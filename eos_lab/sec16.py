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

def _shuffle_idx(N, seed):                                   # Fisher–Yates permutation of range(N) — IDENTICAL across backends (mulberry32)
    rng = mulberry32(u32(int(seed)))
    idx = list(range(N))
    for i in range(N - 1, 0, -1):
        j = int(rng() * (i + 1))
        idx[i], idx[j] = idx[j], idx[i]
    return idx

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

def _eig_ramp(T):                                            # distinct per-index diagonal ramp → breaks the repeated-eigenvalue
    n = T.shape[0]                                           # degeneracy that makes LAPACK syevd fail (code 22) on flat tridiagonals
    sc = max(float(T.abs().max()), 1e-12) * 1e-10
    return T + torch.diag(torch.arange(n, dtype=T.dtype, device=T.device) * sc)

def _safe_eigvalsh(T):                                       # robust symmetric-eigenvalue solve (cifar2/cifar10 §16/§17 near-zero tridiagonals)
    if not bool(torch.isfinite(T).all()):
        T = torch.nan_to_num(T)
    try:
        return torch.linalg.eigvalsh(T)
    except torch._C._LinAlgError:
        Tj = _eig_ramp(T)
        try:
            return torch.linalg.eigvalsh(Tj)
        except torch._C._LinAlgError:
            import numpy as np
            return torch.tensor(np.linalg.eigvalsh(Tj.detach().cpu().numpy()), dtype=T.dtype, device=T.device)

def _safe_eigh(T):
    if not bool(torch.isfinite(T).all()):
        T = torch.nan_to_num(T)
    try:
        return torch.linalg.eigh(T)
    except torch._C._LinAlgError:
        Tj = _eig_ramp(T)
        try:
            return torch.linalg.eigh(Tj)
        except torch._C._LinAlgError:
            import numpy as np
            w, V = np.linalg.eigh(Tj.detach().cpu().numpy())
            return (torch.tensor(w, dtype=T.dtype, device=T.device), torch.tensor(V, dtype=T.dtype, device=T.device))

def _lanc16_vals(hvp, p, n, m, seed, dev, dt):                # top-n & bottom-n eigenVALUES
    _, T, k = _lanc16_core(hvp, p, m, seed, dev, dt); mu = torch.sort(_safe_eigvalsh(T), descending=True).values
    return [float(mu[min(i, k - 1)]) for i in range(n)], [float(mu[max(0, k - 1 - i)]) for i in range(n)]

def _lanc16_ext(hvp, p, n, m, seed, dev, dt):                # top-n & bottom-n (eigVEC, eigVAL) pairs
    Qm, T, k = _lanc16_core(hvp, p, m, seed, dev, dt); mu, Sv = _safe_eigh(T); idx = torch.argsort(mu, descending=True)
    ritz = lambda c: Sv[:, c].to(dtype=dt, device=Qm.device) @ Qm   # Sv (from eigh of CPU T) → Qm's device for the matmul
    tv = [ritz(int(idx[min(i, k - 1)])) for i in range(n)]; bv = [ritz(int(idx[max(0, k - 1 - i)])) for i in range(n)]
    tval = [float(mu[int(idx[min(i, k - 1)])]) for i in range(n)]; bval = [float(mu[int(idx[max(0, k - 1 - i)])]) for i in range(n)]
    return tv, bv, tval, bval


def sec16_run(model, loss, th0, X, Y, lr, warmup, iters, neig, mlan, seed, tol, bgrid=0.1, ares=0.01,
              Xtest=None, Ytest=None, mode="eig", kdir=8):
    # Per iteration: project the positive-residual gradient g₊ = Σ_{r>0} r_k∇f_k onto the TOP-kdir eigenvectors of
    # H̄ and the negative-residual g₋ onto the BOTTOM-kdir, using the plain projection coefficient ⟨g,u_i⟩ (NO σ/η
    # weighting):   proj_pos = Σ_i ⟨g₊,u_i⟩ u_i ,  proj_neg = Σ_i ⟨g₋,u₋i⟩ u₋i ;  then line-search α₁,α₂ on the ±
    # sets, the (β,s) grid on the full loss, and advance by u_sec16 = s·(β·α₁·proj_pos + (1−β)·α₂·proj_neg).
    # EVERY trajectory advances by its own u_sec16; they differ ONLY in how it is built (set by `mode`):
    #   'eig'     (MAIN) top-/bottom-kdir eigvecs of H̄, true ± sets ·
    #   'random'  kdir random unit vecs per side instead of eigvecs · 'shuffle' eigvecs, ± membership permuted ·
    #   'randfix' random dirs FROZEN at warmup-end · 'eigfix' eigvecs FROZEN at warmup-end ·
    #   'gd'      u_sec16 IS the line-searched gradient step t_gd·(1/N)Jᵀr (plain-GD reference baseline).
    # The main run is PURE curvature (no GD fallback; GD lives only in the 'gd' baseline).
    # NB: the main run is PURE curvature — it does NOT use gradient information (GD lives only in the 'gd' baseline).
    dev, dt = th0.device, th0.dtype
    p = model.p
    N = X.shape[0]
    Yf = Y.reshape(-1)
    M = Yf.numel()                                       # EFFECTIVE samples = N·d_out (residual / Jacobian-row dimension)
    Ytf = Ytest.reshape(-1) if Ytest is not None else None
    k = max(1, int(neig))
    mg = max(1, int(round(1.0 / max(1e-9, bgrid))))      # β,s search grid {0,1/mg,…,1} (bgrid = step); finer ⇒ more descent, costs mg² loss-evals
    grid = [i / mg for i in range(mg + 1)]

    def fwd(th):
        return model.forward(th, X).reshape(-1)           # flat (M,) over all N·d_out outputs

    def resid(th):
        return Yf - fwd(th)                                   # y − f(x)  (NOT f−y)  (M,)

    def lossv(th, mask=None):
        out = fwd(th)
        if mask is None:
            return float(loss.value(out, Yf, N))              # full loss normalised by N input samples (matches the main GD)
        n = int(mask.sum())
        return float(loss.value(out[mask], Yf[mask], max(1, n))) if n else 0.0

    def losstest(th):                                         # held-out test loss (None if no test set)
        if Xtest is None:
            return None
        return float(loss.value(model.forward(th, Xtest).reshape(-1), Ytf, Xtest.shape[0]))

    def Hbar_hvp(th):                                         # H̄·v = (1/N) Σ_k ∇²f_k · v
        return lambda v: jac_hvp(model, th, X, v).mean(0)

    def lossH_hvp(th):                                        # ∇²L · v  (central FD of ∇L; matches server.hvpL)
        return lambda v: (grad_loss(model, loss, th + FD_EPS * v, X, Y)[0]
                          - grad_loss(model, loss, th - FD_EPS * v, X, Y)[0]) / (2 * FD_EPS)

    def metrics(th, sd):                                      # loss + test loss + residuals + lossH top-k + funcH top-k/bot-k
        lHt, _ = _lanc16_vals(lossH_hvp(th), p, k, mlan, sd, dev, dt)
        fHt, fHb = _lanc16_vals(Hbar_hvp(th), p, k, mlan, sd + 7, dev, dt)
        return {"L": lossv(th), "Lt": losstest(th), "res": resid(th).detach().cpu().tolist(),
                "lH": lHt, "fHt": fHt, "fHb": fHb}

    def pack(real, look=None):                               # build a streamed record from look-ahead metric dicts
        keys = ("pos", "neg", "mean", "beta")
        d = look if look is not None else {}
        g = lambda kk, f: (f(d[kk]) if kk in d else None)
        rec = {"L": {kk: g(kk, lambda x: x["L"]) for kk in keys},
               "Lt": {kk: g(kk, lambda x: x["Lt"]) for kk in keys},
               "lH": {kk: g(kk, lambda x: x["lH"]) for kk in keys},
               "fH": {kk: g(kk, lambda x: {"top": x["fHt"], "bot": x["fHb"]}) for kk in keys},
               "res": {kk: g(kk, lambda x: x["res"]) for kk in keys}}
        rec["L"]["beta"] = real["L"]; rec["Lt"]["beta"] = real["Lt"]; rec["lH"]["beta"] = real["lH"]
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
    froz = [None, None]                                       # frozen (u₁,u₋₁) for 'randfix'/'eigfix' (captured at warmup-end)
    for it in range(int(warmup) + int(iters)):
        sd = (seed + 1000 * it) & 0x7FFFFFFF
        if it < warmup:                                       # ── standard GD warmup ──
            real = metrics(th, sd)
            J = jac_cols(model, th, X)[0]
            upd = (lr / N) * (J.t() @ resid(th))              # the warmup GD step θ ← θ + (lr/N)·Jᵀr (r = y−f; ADDED)
            yield {"i": it, "phase": "gd", "apos": None, "aneg": None, "beta": None, "scale": None, "tgd": None,
                   "loss": real["L"],                          # Panel 6: ‖update‖₂ — only the real (beta) path exists during warmup
                   "unorm": {"pos": None, "neg": None, "mean": None, "beta": float(upd.norm())}, **pack(real)}
            th = th + upd
            continue
        # ── §16 iteration ──
        r = resid(th)
        J = jac_cols(model, th, X)[0]                         # rows ∇f_k
        if mode == "gd":                                     # GD baseline: u_sec16 IS the line-searched gradient step
            d_gd = (J.t() @ r) / N                            # gradient-descent direction (1/N)Jᵀr, r=y−f
            t_gd = line_search(th, d_gd, None)
            u_sec16 = t_gd * d_gd
            u_pos = u_neg = u_mean = u_sec16                  # the gradient direction has no ± decomposition
            a_pos = a_neg = beta = scale = t_gd
            u_beta = u_sec16
            look = {"pos": metrics(th + u_pos, sd + 1), "neg": metrics(th + u_neg, sd + 2),
                    "mean": metrics(th + u_mean, sd + 3), "beta": metrics(th + u_beta, sd + 4)}
            yield {"i": it, "phase": "sec16", "apos": a_pos, "aneg": a_neg, "beta": beta, "scale": scale, "tgd": t_gd,
                   "loss": look["beta"]["L"],
                   "unorm": {"pos": float(u_pos.norm()), "neg": float(u_neg.norm()),
                             "mean": float(u_mean.norm()), "beta": float(u_beta.norm())}, **pack(look["beta"], look)}
            th = th + u_beta
            if lossv(th) < tol:
                break
            continue
        kd = max(1, min(int(kdir), p // 2))                  # eigenvectors per side (top-kd & bottom-kd)
        if mode in ("random", "randfix"):                    # kd random unit vecs per side instead of eigvecs
            if mode == "randfix" and froz[0] is not None:    # BASELINE C: reuse the warmup-end frozen random directions
                Ut, Ub = froz[0], froz[1]
            else:                                            # BASELINE A: fresh random vectors each iter (or first 'randfix' iter)
                Ut = [_randvec16(p, (sd ^ 0x16A1 ^ (i * 0x101)) & 0x7FFFFFFF, dev, dt) for i in range(kd)]
                Ub = [_randvec16(p, (sd ^ 0x16B1 ^ (i * 0x101)) & 0x7FFFFFFF, dev, dt) for i in range(kd)]
                Ut = [w / w.norm() for w in Ut]; Ub = [w / w.norm() for w in Ub]
                if mode == "randfix":
                    froz[0], froz[1] = Ut, Ub                # freeze at warmup-end (first §16 iter)
        else:                                                # 'eig' / 'eigfix' / 'shuffle': top-kd & bottom-kd eigvecs of H̄
            if mode == "eigfix" and froz[0] is not None:     # BASELINE D: reuse the warmup-end frozen eigvec subspace
                Ut, Ub = froz[0], froz[1]
            else:
                tv, bv, tval, bval = _lanc16_ext(Hbar_hvp(th), p, kd, mlan, sd, dev, dt)
                Ut, Ub = tv, bv
                if mode == "eigfix":
                    froz[0], froz[1] = Ut, Ub
        if mode == "shuffle":                                # BASELINE B: permute ± set membership (keep counts; each sample keeps its real r_k)
            n_pos = int((r > 0).sum()); n_neg = int((r < 0).sum())
            sidx = _shuffle_idx(M, sd ^ 0x16B0)               # permute ± membership over the M effective samples
            pos = torch.zeros(M, dtype=torch.bool, device=dev); neg = torch.zeros(M, dtype=torch.bool, device=dev)
            if n_pos:
                pos[torch.tensor(sidx[:n_pos], dtype=torch.long, device=dev)] = True
            if n_neg:
                neg[torch.tensor(sidx[n_pos:n_pos + n_neg], dtype=torch.long, device=dev)] = True
        else:
            pos, neg = r > 0, r < 0
        g_pos = (J[pos].t() @ r[pos]) if bool(pos.any()) else torch.zeros(p, dtype=dt, device=dev)
        g_neg = (J[neg].t() @ r[neg]) if bool(neg.any()) else torch.zeros(p, dtype=dt, device=dev)
        proj_pos = torch.zeros(p, dtype=dt, device=dev)       # Σ_i ⟨g₊,u_i⟩ u_i  (projection of g₊ onto the top-kd eigvecs; NO σ/η)
        proj_neg = torch.zeros(p, dtype=dt, device=dev)       # Σ_i ⟨g₋,u₋i⟩ u₋i (projection of g₋ onto the bottom-kd eigvecs; NO σ/η)
        for i in range(kd):
            proj_pos = proj_pos + float(g_pos @ Ut[i]) * Ut[i]
            proj_neg = proj_neg + float(g_neg @ Ub[i]) * Ub[i]
        a_pos = line_search(th, proj_pos, pos)
        a_neg = line_search(th, proj_neg, neg)
        u_pos, u_neg = a_pos * proj_pos, a_neg * proj_neg
        u_mean = 0.5 * (u_pos + u_neg)
        beta, scale = min(((bb, ss) for bb in grid for ss in grid),
                          key=lambda bs: lossv(th + bs[1] * (bs[0] * u_pos + (1 - bs[0]) * u_neg)))
        u_sec16 = scale * (beta * u_pos + (1 - beta) * u_neg)
        u_beta = u_sec16; tgd = 0.0                           # advance by u_sec16 — the main run is PURE curvature (no GD fallback)
        look = {"pos": metrics(th + u_pos, sd + 1), "neg": metrics(th + u_neg, sd + 2),
                "mean": metrics(th + u_mean, sd + 3), "beta": metrics(th + u_beta, sd + 4)}
        yield {"i": it, "phase": "sec16", "apos": a_pos, "aneg": a_neg, "beta": beta, "scale": scale, "tgd": tgd,
               "loss": look["beta"]["L"],                       # Panel 6: ‖u_•‖₂ of each look-ahead update
               "unorm": {"pos": float(u_pos.norm()), "neg": float(u_neg.norm()),
                         "mean": float(u_mean.norm()), "beta": float(u_beta.norm())}, **pack(look["beta"], look)}
        th = th + u_beta                                      # ADVANCE (added); non-increasing (s=0 available)
        if lossv(th) < tol:
            break


def sec16_chebyshev_testset(N, degree, dev, dt):
    """Held-out chebyshev test set for §16 Panel 5: the N-1 MIDPOINTS of the training grid linspace(-1,1,N),
    labelled by T_degree (genuinely held out from training). MIRRORS server / index.html."""
    xt = torch.linspace(-1.0, 1.0, max(int(N), 1), dtype=dt, device=dev)
    xm = (xt[:-1] + xt[1:]) / 2.0 if int(N) > 1 else xt.clone()
    if int(degree) <= 0:
        y = torch.ones_like(xm)
    else:
        tm2, tm1 = torch.ones_like(xm), xm.clone()
        for _ in range(2, int(degree) + 1):
            tm2, tm1 = tm1, 2 * xm * tm1 - tm2
        y = tm1
    return xm.unsqueeze(1), y.unsqueeze(1)


def sec16_chebyshev2_testset(N, degree, dev, dt):
    """Held-out chebyshev-2 test set for §16 Panel 5: the N-1 MIDPOINTS of linspace(-1,1,N), labelled by
    T_degree=cos(degree·arccos x) (the closed form). MIRRORS server._sec16_chebyshev2_testset / index.html sec16Cheb2."""
    xt = torch.linspace(-1.0, 1.0, max(int(N), 1), dtype=dt, device=dev)
    xm = (xt[:-1] + xt[1:]) / 2.0 if int(N) > 1 else xt.clone()
    y = torch.cos(float(int(degree)) * torch.arccos(torch.clamp(xm, -1.0, 1.0)))
    return xm.unsqueeze(1), y.unsqueeze(1)


def sec16_driver(model, loss, th0, X, Y, lr, warmup, iters, neig, mlan, seed, tol, bgrid=0.1, ares=0.01,
                 Xtest=None, Ytest=None, baselines=False, kdir=8):
    """Run §16 (mode='eig', the curvature step) and, if `baselines`, FIVE extra trajectories — 'random' (A),
    'shuffle' (B), 'randfix' (C: frozen random dirs), 'eigfix' (D: frozen H̄ eigvec dirs) and 'gd'
    (E: the plain gradient-descent step) — in lockstep from the SAME θ₀. All six advance by their own u_sec16;
    they differ ONLY in how u_sec16 is built. Metrics attach as rec['bA']/['bB']/['bC']/['bD']/['bE']."""
    gmain = sec16_run(model, loss, th0, X, Y, lr, warmup, iters, neig, mlan, seed, tol, bgrid, ares, Xtest, Ytest, "eig", kdir)
    if not baselines:
        yield from gmain
        return
    # baselines never early-stop (tol=-1) so all trajectories stay the same length (lockstep)
    mk = lambda md: sec16_run(model, loss, th0, X, Y, lr, warmup, iters, neig, mlan, seed, -1.0, bgrid, ares, Xtest, Ytest, md, kdir)
    gA, gB, gC, gD, gE = mk("random"), mk("shuffle"), mk("randfix"), mk("eigfix"), mk("gd")
    sub = lambda r: {kk: r[kk] for kk in ("L", "Lt", "lH", "fH", "res", "unorm")}
    for rec in gmain:
        rec = dict(rec)
        rec["bA"] = sub(next(gA)); rec["bB"] = sub(next(gB))
        rec["bC"] = sub(next(gC)); rec["bD"] = sub(next(gD)); rec["bE"] = sub(next(gE))
        yield rec
