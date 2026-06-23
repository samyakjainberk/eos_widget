"""§17 — PER-SAMPLE curvature-aligned per-residual-sign optimizer (standalone), MIRRORS server._sec17_run /
index.html sec17Run.  IDENTICAL to §16 EXCEPT the curvature directions are PER-SAMPLE:

  §16 used ONE shared pair from the AVERAGED function Hessian H̄ = (1/N)Σ_k ∇²f_k — top eigvec u₁ for every
  positive-residual sample, bottom eigvec u₋₁ for every negative one.

  §17 instead uses EACH SAMPLE's OWN function Hessian  Q_k = ∇²f_k:
    * a POSITIVE-residual sample k (r_k>0) projects onto the TOP (most-positive) eigvec u_k⁺ of its own Q_k,
    * a NEGATIVE-residual sample k (r_k<0) projects onto the BOTTOM (most-negative) eigvec u_k⁻ of its own Q_k,
  so the two curvature steps accumulate per-sample contributions (each scaled by that sample's own eigenvalue):
      g₊-step  proj_pos = Σ_{r_k>0} r_k·⟨∇f_k, u_k⁺⟩·σ_k⁺·u_k⁺ ,
      g₋-step  proj_neg = Σ_{r_k<0} r_k·⟨∇f_k, u_k⁻⟩·σ_k⁻·u_k⁻ .
  Everything downstream (α_pos/α_neg golden-section line search, the (β,s) grid, the GD-best-of advance, and the
  four look-ahead metric panels) is byte-for-byte the §16 machinery.  Per-sample Q_k·v is the FD Jacobian-of-grad
  (jac_hvp(...)[k]) — the SAME primitive §16 averages for H̄ — so picking row k gives Q_k·v with no new numerics.

The five baselines mirror §16 exactly but per-sample: A='random' (per-sample random unit dirs, keep σ_k), B='shuffle'
(real per-sample eigvecs but ± set membership permuted), C='randfix' (per-sample random dirs FROZEN at warmup-end),
D='eigfix' (per-sample Q_k eigvecs FROZEN at warmup-end).  σ_k⁺/σ_k⁻ are ALWAYS recomputed each iteration.

The diagnostic panels (loss, loss-Hessian ∇²L eig, function-Hessian H̄ eig, residuals, test loss, ‖update‖) are
the SAME observables as §16 (so the two sections are directly comparable); only the UPDATE direction is per-sample.
"""
import math
import torch

from .models import jac_cols, jac_hvp, grad_loss, FD_EPS
from .rng import mulberry32, u32, gauss
from .sec16 import _randvec16, _shuffle_idx, _lanc16_vals, _lanc16_ext


def sec17_run(model, loss, th0, X, Y, lr, warmup, iters, neig, mlan, seed, tol, bgrid=0.1, ares=0.01,
              Xtest=None, Ytest=None, mode="eig", use_gd=True):
    # mode: 'eig' (per-sample top/bottom eigvecs of Q_k, true ± sets) · 'random' (per-sample random unit vecs, keep σ_k) ·
    #       'shuffle' (per-sample eigvecs, ± set membership permuted) · 'randfix' (per-sample random dirs FROZEN at warmup-end) ·
    #       'eigfix' (per-sample Q_k eigvecs FROZEN at warmup-end).  use_gd: best{curvature,GD} (main) vs pure curvature (baselines).
    dev, dt = th0.device, th0.dtype
    p = model.p
    N = X.shape[0]
    Yf = Y.reshape(-1)
    M = Yf.numel()                                       # EFFECTIVE samples = N·d_out (residual / Jacobian-row / per-sample-Q dimension)
    Ytf = Ytest.reshape(-1) if Ytest is not None else None
    k = max(1, int(neig))
    mg = max(1, int(round(1.0 / max(1e-9, bgrid))))
    grid = [i / mg for i in range(mg + 1)]

    def fwd(th):
        return model.forward(th, X).reshape(-1)

    def resid(th):
        return Yf - fwd(th)

    def lossv(th, mask=None):
        out = fwd(th)
        if mask is None:
            return float(loss.value(out, Yf, N))
        n = int(mask.sum())
        return float(loss.value(out[mask], Yf[mask], max(1, n))) if n else 0.0

    def losstest(th):
        if Xtest is None:
            return None
        return float(loss.value(model.forward(th, Xtest).reshape(-1), Ytf, Xtest.shape[0]))

    def Hbar_hvp(th):                                        # H̄·v = (1/N) Σ_k ∇²f_k · v  (Panel 3 diagnostic, same as §16)
        return lambda v: jac_hvp(model, th, X, v).mean(0)

    def Qk_hvp(th, kk):                                      # Q_k·v = ∇²f_k · v  (per-sample; row k of the FD Jacobian-of-grad)
        return lambda v: jac_hvp(model, th, X, v)[kk]

    def lossH_hvp(th):
        return lambda v: (grad_loss(model, loss, th + FD_EPS * v, X, Y)[0]
                          - grad_loss(model, loss, th - FD_EPS * v, X, Y)[0]) / (2 * FD_EPS)

    def metrics(th, sd):                                     # IDENTICAL to §16 (loss/test/residuals + lossH top-k + H̄ top/bot-k)
        lHt, _ = _lanc16_vals(lossH_hvp(th), p, k, mlan, sd, dev, dt)
        fHt, fHb = _lanc16_vals(Hbar_hvp(th), p, k, mlan, sd + 7, dev, dt)
        return {"L": lossv(th), "Lt": losstest(th), "res": resid(th).detach().cpu().tolist(),
                "lH": lHt, "fHt": fHt, "fHb": fHb}

    def pack(real, look=None):                              # IDENTICAL to §16
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
        al = math.floor((a + b) / 2.0 / ares + 0.5) * ares
        return al if lossv(th + al * direction, mask) <= lossv(th, mask) else 0.0

    th = th0.detach().clone()
    froz = [None, None]                                      # frozen per-sample [Up(list of M vecs), Um] for 'randfix'/'eigfix'
    for it in range(int(warmup) + int(iters)):
        sd = (seed + 1000 * it) & 0x7FFFFFFF
        if it < warmup:                                      # ── standard GD warmup (identical to §16) ──
            real = metrics(th, sd)
            J = jac_cols(model, th, X)[0]
            upd = (lr / N) * (J.t() @ resid(th))
            yield {"i": it, "phase": "gd", "apos": None, "aneg": None, "beta": None, "scale": None, "tgd": None,
                   "loss": real["L"],
                   "unorm": {"pos": None, "neg": None, "mean": None, "beta": float(upd.norm())}, **pack(real)}
            th = th + upd
            continue
        # ── §17 iteration: per-sample curvature directions ──
        r = resid(th)
        J = jac_cols(model, th, X)[0]                        # rows ∇f_k
        psd = (sd ^ 0x17C0FFEE) & 0x7FFFFFFF                 # per-sample Lanczos/random seed base (disjoint from metrics seeds)
        Up = []; Um = []; Sp = []; Sm = []                   # per-sample top/bottom eigvec + eigenvalue
        for kk in range(M):
            skk = (psd + kk * 1009) & 0x7FFFFFFF
            hk = Qk_hvp(th, kk)
            if mode in ("random", "randfix"):               # per-sample random dirs, keep σ_k⁺/σ_k⁻ of Q_k
                tH, bH = _lanc16_vals(hk, p, 1, mlan, skk, dev, dt)
                Sp.append(tH[0]); Sm.append(bH[0])
                up = _randvec16(p, (skk ^ 0x17A1) & 0x7FFFFFFF, dev, dt); up = up / up.norm()
                um = _randvec16(p, (skk ^ 0x17A2) & 0x7FFFFFFF, dev, dt); um = um / um.norm()
                Up.append(up); Um.append(um)
            else:                                           # 'eig'/'eigfix'/'shuffle': real per-sample top/bottom eigenpairs of Q_k
                tv, bv, tval, bval = _lanc16_ext(hk, p, 1, mlan, skk, dev, dt)
                Sp.append(tval[0]); Sm.append(bval[0])
                Up.append(tv[0]); Um.append(bv[0])
        if mode in ("randfix", "eigfix"):                   # BASELINES C/D: freeze the per-sample direction set at warmup-end (σ still recomputed)
            if froz[0] is None:
                froz[0] = [u.clone() for u in Up]; froz[1] = [u.clone() for u in Um]
            Up, Um = froz[0], froz[1]
        if mode == "shuffle":                               # BASELINE B: permute ± set membership over the M effective samples (counts preserved)
            n_pos = int((r > 0).sum()); n_neg = int((r < 0).sum())
            sidx = _shuffle_idx(M, sd ^ 0x16B0)
            pos = torch.zeros(M, dtype=torch.bool, device=dev); neg = torch.zeros(M, dtype=torch.bool, device=dev)
            if n_pos:
                pos[torch.tensor(sidx[:n_pos], dtype=torch.long, device=dev)] = True
            if n_neg:
                neg[torch.tensor(sidx[n_pos:n_pos + n_neg], dtype=torch.long, device=dev)] = True
        else:
            pos, neg = r > 0, r < 0
        proj_pos = torch.zeros(p, dtype=dt, device=dev)     # Σ_{r_k>0} r_k·⟨∇f_k, u_k⁺⟩·σ_k⁺·u_k⁺
        proj_neg = torch.zeros(p, dtype=dt, device=dev)     # Σ_{r_k<0} r_k·⟨∇f_k, u_k⁻⟩·σ_k⁻·u_k⁻
        for kk in range(M):
            if bool(pos[kk]):
                proj_pos = proj_pos + (float(r[kk]) * float(J[kk] @ Up[kk]) * Sp[kk]) * Up[kk]
            elif bool(neg[kk]):
                proj_neg = proj_neg + (float(r[kk]) * float(J[kk] @ Um[kk]) * Sm[kk]) * Um[kk]
        a_pos = line_search(th, proj_pos, pos)
        a_neg = line_search(th, proj_neg, neg)
        u_pos, u_neg = a_pos * proj_pos, a_neg * proj_neg
        u_mean = 0.5 * (u_pos + u_neg)
        beta, scale = min(((bb, ss) for bb in grid for ss in grid),
                          key=lambda bs: lossv(th + bs[1] * (bs[0] * u_pos + (1 - bs[0]) * u_neg)))
        u_sec = scale * (beta * u_pos + (1 - beta) * u_neg)
        if use_gd:
            d_gd = (J.t() @ resid(th)) / N
            t_gd = line_search(th, d_gd, None); u_gd = t_gd * d_gd
            if lossv(th + u_gd) < lossv(th + u_sec):
                u_beta = u_gd; tgd = t_gd
            else:
                u_beta = u_sec; tgd = 0.0
        else:
            u_beta = u_sec; tgd = 0.0
        look = {"pos": metrics(th + u_pos, sd + 1), "neg": metrics(th + u_neg, sd + 2),
                "mean": metrics(th + u_mean, sd + 3), "beta": metrics(th + u_beta, sd + 4)}
        yield {"i": it, "phase": "sec17", "apos": a_pos, "aneg": a_neg, "beta": beta, "scale": scale, "tgd": tgd,
               "loss": look["beta"]["L"],
               "unorm": {"pos": float(u_pos.norm()), "neg": float(u_neg.norm()),
                         "mean": float(u_mean.norm()), "beta": float(u_beta.norm())}, **pack(look["beta"], look)}
        th = th + u_beta
        if lossv(th) < tol:
            break


def sec17_driver(model, loss, th0, X, Y, lr, warmup, iters, neig, mlan, seed, tol, bgrid=0.1, ares=0.01,
                 Xtest=None, Ytest=None, baselines=False):
    """Run §17 (mode='eig') and, if `baselines`, FOUR extra trajectories — 'random' (A), 'shuffle' (B),
    'randfix' (C), 'eigfix' (D) — in lockstep from the SAME θ₀, attaching per-iteration metrics as
    rec['bA']/['bB']/['bC']/['bD'].  MIRRORS sec16_driver."""
    gmain = sec17_run(model, loss, th0, X, Y, lr, warmup, iters, neig, mlan, seed, tol, bgrid, ares, Xtest, Ytest, "eig", True)
    if not baselines:
        yield from gmain
        return
    mk = lambda md: sec17_run(model, loss, th0, X, Y, lr, warmup, iters, neig, mlan, seed, -1.0, bgrid, ares, Xtest, Ytest, md, False)
    gA, gB, gC, gD = mk("random"), mk("shuffle"), mk("randfix"), mk("eigfix")
    sub = lambda r: {kk: r[kk] for kk in ("L", "Lt", "lH", "fH", "res", "unorm")}
    for rec in gmain:
        rec = dict(rec)
        rec["bA"] = sub(next(gA)); rec["bB"] = sub(next(gB))
        rec["bC"] = sub(next(gC)); rec["bD"] = sub(next(gD))
        yield rec
