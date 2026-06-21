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
      - (β,s) = argmin over β∈{0..1}×s∈{0..1} (step 0.1) of the FULL loss at θ + s·(β·u_pos+(1−β)·u_neg)
                (s=0 ⇒ no-op, so the ADVANCING step never increases the full loss),  u_beta = s·(β·u_pos+(1−β)·u_neg),
      - the real path ADVANCES by u_beta (θ ← θ + u_beta — ADDED).
  Repeats until the loss is tiny or the iteration cap is hit.

Per iteration we record, at the FOUR look-ahead points {θ+u_pos, θ+u_neg, θ+u_mean, θ+u_beta}:
  panel 1 — full loss (+ α_pos,α_neg,β,s as plot 5);  panel 2 — top-k eigvals of the LOSS Hessian ∇²L;
  panel 3 — top-k & bottom-k eigvals of the function Hessian H̄;  panel 4 — per-sample residuals (y−f).
During the warmup phase only the real (GD) path exists ⇒ the pos/neg/mean entries are None.
"""
import math
import torch

from .models import jac_cols, jac_hvp, grad_loss, FD_EPS
from .linalg import lanczos_extreme, lanczos_extreme_vals


def sec16_run(model, loss, th0, X, Y, lr, warmup, iters, neig, mlan, seed, tol):
    dev, dt = th0.device, th0.dtype
    p = model.p
    N = X.shape[0]
    Yf = Y.reshape(-1)
    k = max(1, int(neig))
    grid = [i / 10.0 for i in range(11)]

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
        lHt, _ = lanczos_extreme_vals(lossH_hvp(th), p, k, mlan, sd, dev, dt)
        fHt, fHb = lanczos_extreme_vals(Hbar_hvp(th), p, k, mlan, sd + 7, dev, dt)
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

    def line_search(th, direction, mask, lo=0.0, hi=10.0, ls_tol=0.01):
        gr = (math.sqrt(5) - 1) / 2
        a, b = lo, hi
        c = b - gr * (b - a); dd = a + gr * (b - a)
        fc = lossv(th + c * direction, mask); fdd = lossv(th + dd * direction, mask)
        while (b - a) > ls_tol:
            if fc < fdd:
                b, dd, fdd = dd, c, fc; c = b - gr * (b - a); fc = lossv(th + c * direction, mask)
            else:
                a, c, fc = c, dd, fdd; dd = a + gr * (b - a); fdd = lossv(th + dd * direction, mask)
        al = round((a + b) / 2.0, 2)
        return al if lossv(th + al * direction, mask) <= lossv(th, mask) else 0.0   # α≥0; never worse than α=0

    th = th0.detach().clone()
    for it in range(int(warmup) + int(iters)):
        sd = (seed + 1000 * it) & 0x7FFFFFFF
        if it < warmup:                                       # ── standard GD warmup ──
            real = metrics(th, sd)
            yield {"i": it, "phase": "gd", "apos": None, "aneg": None, "beta": None, "scale": None,
                   "loss": real["L"], **pack(real)}
            J = jac_cols(model, th, X)[0]
            th = th + (lr / N) * (J.t() @ resid(th))          # θ ← θ + (lr/N)·Jᵀr   (r = y−f; ADDED)
            continue
        # ── §16 iteration ──
        r = resid(th)
        J = jac_cols(model, th, X)[0]                         # rows ∇f_k
        tv, bv, tval, bval = lanczos_extreme(Hbar_hvp(th), p, k, mlan, sd, dev, dt)
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
        u_beta = scale * (beta * u_pos + (1 - beta) * u_neg)
        look = {"pos": metrics(th + u_pos, sd + 1), "neg": metrics(th + u_neg, sd + 2),
                "mean": metrics(th + u_mean, sd + 3), "beta": metrics(th + u_beta, sd + 4)}
        yield {"i": it, "phase": "sec16", "apos": a_pos, "aneg": a_neg, "beta": beta, "scale": scale,
               "loss": look["beta"]["L"], **pack(look["beta"], look)}
        th = th + u_beta                                      # ADVANCE (added); non-increasing (s=0 available)
        if lossv(th) < tol:
            break
