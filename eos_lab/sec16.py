"""§16 — curvature-aligned per-residual-sign optimizer (standalone), MIRRORS server._sec16_run / index.html sec16Run.

Standalone iterative optimizer started from θ₀:
  * first `warmup` STANDARD GD steps (θ ← θ + (lr/N)·Jᵀr, r = y − f(x); the update is ADDED),
  * then switch to the §16 update: at each iteration, around the current θ,
      - H̄ = (1/N) Σ_k ∇²f_k  (function Hessian averaged over samples); top (eigval σ₁, eigvec u₁) and
        bottom (eigval σ₋₁, eigvec u₋₁) pairs (Lanczos),
      - POSITIVE-residual samples (r>0):  g₊ = Σ_{r_k>0} r_k ∇f_k ;  proj_u1   = ⟨g₊,u₁⟩·σ₁·u₁ ,
      - NEGATIVE-residual samples (r<0):  g₋ = Σ_{r_k<0} r_k ∇f_k ;  proj_u(-1) = ⟨g₋,u₋₁⟩·σ₋₁·u₋₁ ,  (each scaled by its eigenvalue)
      - α_pos = argmin_{α∈[0,10]}  loss on the POSITIVE set  along  θ + α·proj_u1   (golden-section, res 0.01, α≥0),
      - α_neg = argmin_{α∈[0,10]}  loss on the NEGATIVE set  along  θ + α·proj_u(-1),
      - u_pos = α_pos·proj_u1 ,  u_neg = α_neg·proj_u(-1) ,  u_mean = ½(u_pos+u_neg),
      - (β,s) = argmin over β∈{0..1}×s∈{0..1} (step 0.1) of the FULL loss at θ + s·(β·u_pos+(1−β)·u_neg) — the §16 curvature step,
      - a GD step u_gd = t_gd·(1/N)Jᵀr with t_gd full-loss line-searched (always a descent direction),
      - the real path ADVANCES by whichever of {§16 step, GD step} lowers the FULL loss more (θ ← θ + u_beta — ADDED), so the
        loss STRICTLY DECREASES every iteration: the curvature step is taken when it beats GD (recorded tgd=0), else GD (tgd=t_gd).
  Repeats until the loss is tiny or the iteration cap is hit.

Per iteration we record, at the FOUR look-ahead points {θ+u_pos, θ+u_neg, θ+u_mean, θ+u_beta}:
  panel 1 — full loss (+ α_pos,α_neg,β,s as plot 5);  panel 2 — top-k eigvals of the LOSS Hessian ∇²L;
  panel 3 — top-k & bottom-k eigvals of the function Hessian H̄;  panel 4 — per-sample residuals (y−f).
During the warmup phase only the real (GD) path exists ⇒ the pos/neg/mean entries are None.
"""
import math
import torch

from .models import jac_cols, jac_hvp, grad_loss, FD_EPS
from .rng import mulberry32, u32, gauss


# §16 uses a mulberry32+gauss Lanczos start vector — IDENTICAL across server/eos_lab/browser (the same
# primitive that makes θ₀ cross-backend-identical). H̄'s spectrum is clustered, so at finite Lanczos m the
# eigenpair estimate is start-vector-dependent; sharing the start makes the §16 trajectory bit-consistent
# across all three backends (the shared torch-RNG Lanczos in linalg.py would NOT match the browser).
def _randvec16(p, seed, dev, dt):
    rng = mulberry32(u32(int(seed)))
    return torch.tensor([gauss(rng) for _ in range(p)], dtype=dt, device=dev)

def _lanc16_core(hvp, p, m, seed, dev, dt):
    q = _randvec16(p, seed, dev, dt); q = q / q.norm()
    Q = []; al = []; be = []
    qp = torch.zeros(p, dtype=dt, device=dev); beta = torch.zeros((), dtype=dt, device=dev)
    for _ in range(min(p, m)):
        Q.append(q); w = hvp(q); a = (w @ q); al.append(float(a)); w = w - a * q - beta * qp
        Qm = torch.stack(Q)
        for _ in range(2):
            w = w - Qm.t() @ (Qm @ w)
        beta = w.norm(); be.append(float(beta))
        if float(beta) < 1e-10:
            break
        qp = q; q = w / beta
    k = len(Q); T = torch.zeros(k, k, dtype=torch.float64)
    for i in range(k):
        T[i, i] = al[i]
    for i in range(k - 1):
        T[i, i + 1] = be[i]; T[i + 1, i] = be[i]
    return torch.stack(Q), T, k

def _lanc16_vals(hvp, p, n, m, seed, dev, dt):                # top-n & bottom-n eigenVALUES
    _, T, k = _lanc16_core(hvp, p, m, seed, dev, dt); mu = torch.sort(torch.linalg.eigvalsh(T), descending=True).values
    return [float(mu[min(i, k - 1)]) for i in range(n)], [float(mu[max(0, k - 1 - i)]) for i in range(n)]

def _lanc16_ext(hvp, p, n, m, seed, dev, dt):                # top-n & bottom-n (eigVEC, eigVAL) pairs
    Qm, T, k = _lanc16_core(hvp, p, m, seed, dev, dt); mu, Sv = torch.linalg.eigh(T); idx = torch.argsort(mu, descending=True)
    ritz = lambda c: Sv[:, c].to(dtype=dt, device=Qm.device) @ Qm   # Sv (from eigh of CPU T) → Qm's device for the matmul
    tv = [ritz(int(idx[min(i, k - 1)])) for i in range(n)]; bv = [ritz(int(idx[max(0, k - 1 - i)])) for i in range(n)]
    tval = [float(mu[int(idx[min(i, k - 1)])]) for i in range(n)]; bval = [float(mu[int(idx[max(0, k - 1 - i)])]) for i in range(n)]
    return tv, bv, tval, bval


def sec16_run(model, loss, th0, X, Y, lr, warmup, iters, neig, mlan, seed, tol, bgrid=0.1, ares=0.01):
    dev, dt = th0.device, th0.dtype
    p = model.p
    N = X.shape[0]
    Yf = Y.reshape(-1)
    k = max(1, int(neig))
    mg = max(1, int(round(1.0 / max(1e-9, bgrid))))      # β,s search grid {0,1/mg,…,1} (bgrid = step); finer ⇒ more descent, costs mg² loss-evals
    grid = [i / mg for i in range(mg + 1)]

    def fwd(th):
        return model.forward(th, X).reshape(-1)

    def resid(th):
        return Yf - fwd(th)                                   # y − f(x)  (NOT f−y)

    def lossv(th, mask=None):
        out = fwd(th)
        if mask is None:
            return float(loss.value(out, Yf, out.numel()))
        n = int(mask.sum())
        return float(loss.value(out[mask], Yf[mask], max(1, n))) if n else 0.0

    def Hbar_hvp(th):                                         # H̄·v = (1/N) Σ_k ∇²f_k · v
        return lambda v: jac_hvp(model, th, X, v).mean(0)

    def lossH_hvp(th):                                        # ∇²L · v  (central FD of ∇L; matches server.hvpL)
        return lambda v: (grad_loss(model, loss, th + FD_EPS * v, X, Y)[0]
                          - grad_loss(model, loss, th - FD_EPS * v, X, Y)[0]) / (2 * FD_EPS)

    def metrics(th, sd):                                      # loss + residuals + lossH top-k + funcH top-k/bot-k
        lHt, _ = _lanc16_vals(lossH_hvp(th), p, k, mlan, sd, dev, dt)
        fHt, fHb = _lanc16_vals(Hbar_hvp(th), p, k, mlan, sd + 7, dev, dt)
        return {"L": lossv(th), "res": resid(th).detach().cpu().tolist(),
                "lH": lHt, "fHt": fHt, "fHb": fHb}

    def pack(real, look=None):                               # build a streamed record from look-ahead metric dicts
        keys = ("pos", "neg", "mean", "beta")
        d = look if look is not None else {}
        getL = lambda kk: (d[kk]["L"] if kk in d else None)
        getlH = lambda kk: (d[kk]["lH"] if kk in d else None)
        getfH = lambda kk: ({"top": d[kk]["fHt"], "bot": d[kk]["fHb"]} if kk in d else None)
        getr = lambda kk: (d[kk]["res"] if kk in d else None)
        rec = {"L": {kk: getL(kk) for kk in keys}, "lH": {kk: getlH(kk) for kk in keys},
               "fH": {kk: getfH(kk) for kk in keys}, "res": {kk: getr(kk) for kk in keys}}
        rec["L"]["beta"] = real["L"]; rec["lH"]["beta"] = real["lH"]
        rec["fH"]["beta"] = {"top": real["fHt"], "bot": real["fHb"]}; rec["res"]["beta"] = real["res"]
        return rec

    def line_search(th, direction, mask, lo=0.0, hi=10.0, ls_tol=ares):
        gr = (math.sqrt(5) - 1) / 2
        a, b = lo, hi
        c = b - gr * (b - a); dd = a + gr * (b - a)
        fc = lossv(th + c * direction, mask); fdd = lossv(th + dd * direction, mask)
        while (b - a) > ls_tol:
            if fc < fdd:
                b, dd, fdd = dd, c, fc; c = b - gr * (b - a); fc = lossv(th + c * direction, mask)
            else:
                a, c, fc = c, dd, fdd; dd = a + gr * (b - a); fdd = lossv(th + dd * direction, mask)
        al = math.floor((a + b) / 2.0 / ares + 0.5) * ares     # snap to the `ares` grid (half-up; identical in JS), α≥0
        return al if lossv(th + al * direction, mask) <= lossv(th, mask) else 0.0   # α≥0; never worse than α=0

    th = th0.detach().clone()
    for it in range(int(warmup) + int(iters)):
        sd = (seed + 1000 * it) & 0x7FFFFFFF
        if it < warmup:                                       # ── standard GD warmup ──
            real = metrics(th, sd)
            yield {"i": it, "phase": "gd", "apos": None, "aneg": None, "beta": None, "scale": None, "tgd": None,
                   "loss": real["L"], **pack(real)}
            J = jac_cols(model, th, X)[0]
            th = th + (lr / N) * (J.t() @ resid(th))          # θ ← θ + (lr/N)·Jᵀr   (r = y−f; ADDED)
            continue
        # ── §16 iteration ──
        r = resid(th)
        J = jac_cols(model, th, X)[0]                         # rows ∇f_k
        tv, bv, tval, bval = _lanc16_ext(Hbar_hvp(th), p, k, mlan, sd, dev, dt)
        u1, um1 = tv[0], bv[0]
        sig1, sigm1 = tval[0], bval[0]                        # top/bottom eigenvalues of H̄
        pos, neg = r > 0, r < 0
        g_pos = (J[pos].t() @ r[pos]) if bool(pos.any()) else torch.zeros(p, dtype=dt, device=dev)
        g_neg = (J[neg].t() @ r[neg]) if bool(neg.any()) else torch.zeros(p, dtype=dt, device=dev)
        proj_pos = (float(g_pos @ u1) * sig1) * u1            # ⟨g₊,u₁⟩·σ₁·u₁   (×eigenvalue; sign-invariant in u₁)
        proj_neg = (float(g_neg @ um1) * sigm1) * um1         # ⟨g₋,u₋₁⟩·σ₋₁·u₋₁ (×eigenvalue)
        a_pos = line_search(th, proj_pos, pos)
        a_neg = line_search(th, proj_neg, neg)
        u_pos, u_neg = a_pos * proj_pos, a_neg * proj_neg
        u_mean = 0.5 * (u_pos + u_neg)
        beta, scale = min(((bb, ss) for bb in grid for ss in grid),
                          key=lambda bs: lossv(th + bs[1] * (bs[0] * u_pos + (1 - bs[0]) * u_neg)))
        u_sec16 = scale * (beta * u_pos + (1 - beta) * u_neg)
        # GD step (full-loss line-search along the gradient) — guarantees the advancing loss STRICTLY decreases.
        d_gd = (J.t() @ resid(th)) / N                        # gradient-descent direction (1/N)Jᵀr, r=y−f
        t_gd = line_search(th, d_gd, None)
        u_gd = t_gd * d_gd
        if lossv(th + u_gd) < lossv(th + u_sec16):            # advance by whichever lowers the FULL loss more
            u_beta = u_gd; tgd = t_gd                         #   → loss goes DOWN every iteration (GD when the §16 step doesn't help)
        else:
            u_beta = u_sec16; tgd = 0.0                       #   → the §16 curvature step is used when it beats GD (tgd=0)
        look = {"pos": metrics(th + u_pos, sd + 1), "neg": metrics(th + u_neg, sd + 2),
                "mean": metrics(th + u_mean, sd + 3), "beta": metrics(th + u_beta, sd + 4)}
        yield {"i": it, "phase": "sec16", "apos": a_pos, "aneg": a_neg, "beta": beta, "scale": scale, "tgd": tgd,
               "loss": look["beta"]["L"], **pack(look["beta"], look)}
        th = th + u_beta                                      # ADVANCE (added); non-increasing (s=0 available)
        if lossv(th) < tol:
            break
