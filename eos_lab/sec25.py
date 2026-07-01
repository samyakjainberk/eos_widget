"""§25 — gradient-norm evolution + the product-rule split of d/dt(J·r) + 6-gradient cosine alignments
(standalone payload), MIRRORS server.py's per-step g25 block (server.py:3235-3268) and _sec25_cosines
(server.py:1913-1943) EXACTLY.

Under the gradient-flow velocity θ̇ = −∇L (no lr): J̇ = Q·θ̇ ⇒ J̇·r = −M_r∇L (M_r = Σ_k r_k Q_k);
ṙ = −J·θ̇ ⇒ J·ṙ = (JᵀJ)∇L. The single time-series plot carries:
  gn  = ‖∇L‖
  jdr = ‖J̇·r‖ = ‖M_r ∇L‖                 (M_r∇L via hvp_S — the residual-weighted function-Hessian HVP)
  jrd = ‖J·ṙ‖ = ‖JᵀJ ∇L‖                  (JᵀJ∇L = Jᵀ(J·∇L), exact via jac_cols + grad_loss)
  II  = c²·Σ_k ∇f_{t,k}ᵀ Q_{t-1,k} (Q̃ g₂)  with Q̃ = Σ_j r_{t-1,j} Q_{t-2,j},  g₂ = J_{t-2}ᵀ r_{t-2}
  III = c²·Σ_k ∇f_{t,k}ᵀ Q_{t-1,k} (J_{t-1}ᵀ J_{t-2} g₂)              (c = η/N = lr/N)
  ddJr  = ‖J·ṙ + J̇·r‖₁ = ‖JᵀJ∇L − M_r∇L‖₁ = Σ_i |（JJg25 − Mr25)_i|
  cosJr = cos(J·ṙ, J̇·r) = cos(JᵀJ∇L, −M_r∇L) = −⟨JJg25, Mr25⟩/(‖JJg25‖‖Mr25‖)

II/III are the cross-terms of the ‖J‖²_F (=tr NTK) 2nd-difference decomposition (§15 panel 1); they need
3 CONSECUTIVE GD steps (t, t−1, t−2 in the history) ⇒ None until the rolling buffer is filled. The inline
II/III formula here is BYTE-IDENTICAL to the II/III sub-expressions in eos_lab.linalg.sec15_stats
(`a = rtm1 @ hvp2(g2); II = c²·(Jt·hvp1(a)).sum()`, `z = Jtm1ᵀ(Jtm2 g2); III = c²·(Jt·hvp1(z)).sum()`),
but evaluated at the HISTORY's θ_{t-1}, θ_{t-2} (the §15 path passes the same Jt/Jtm1/Jtm2/rtm1/rtm2 and
hvp1/hvp2 closures bound to those θ's), so we inline it with jac_hvp exactly as the server does.

cosA/cosB/cosF/cosG/gnorms (MLP only) are pairwise cosines of ∇θ of 7 scalars, EXACT autograd both sides:
  φ1 = ‖r‖²   φ2 = ‖J‖²_F (= ∇ tr NTK)   φ3 = ‖∇L‖²   φ4 = ‖(ΣQ_k)·Jᵀr‖²   φ5 = ‖J·Jᵀr‖² (=‖NTK·r‖², TWO J's)   φ6 = ‖f‖₂²   φ7 = ‖ġ‖² (ġ=d(∇L)/dt=−H∇L)
  cosA = [(1,2),(1,3),(1,4),(1,5),(2,3)]   cosB = [(2,4),(2,5),(3,4),(3,5),(4,5)]
  cosF = [cos(6,1),cos(6,2),cos(6,3),cos(6,4),cos(6,5)] (vector 6 vs the other five)
  cosG = [cos(7,1),cos(7,2),cos(7,3),cos(7,4),cos(7,5),cos(7,6)] (vector 7 vs the other six)   gnorms = [‖∇φ_i‖]_{i=1..7}
φ5 squares J·(Jᵀr) — vector 5 is ∇θ‖JJᵀr‖², NOT ∇L; φ6 squares the model output f; φ7 squares ġ=−H∇L (H=∇²L). Mirrors server._sec25_cosines line-by-line.

PARITY NOTE (FD-vs-exact): server's MlpModel uses FINITE-DIFFERENCE HVPs (EPS=1e-3) for hvpS, while eos_lab's
hvp_S is EXACT autograd (double-backward). So the Mr25-derived fields (jdr, and the Mr25 part of ddJr/cosJr)
match server only to the FD truncation error (~1e-6 relative in fp64), the documented FD-vs-exact gap — NOT a
port bug. jrd (no hvpS), II/III (jac_hvp, FD on BOTH sides), and the 7-vector cosA/cosB/cosF/cosG/gnorms (exact autograd
on BOTH sides) match to ~1e-12..1e-13 in fp64.
"""
import torch

from .models import jac_cols, jac_hvp, hvp_S, grad_loss


def _sec25_cosines(model, loss, th, X, Y, N, outD):
    """§25 plots 2/3 (pairwise cosines) + plot 3 (cos of ∇‖f‖² vs the 5) + plot 4 (cos of ∇‖ġ‖² vs the 6) + the 7 ∇θ-vector norms —
    MLP only, exact autograd via torch.func; every scalar is ‖·‖² so the gradient DIRECTION matches the
    user's ‖·‖/‖·‖² forms. MIRRORS server._sec25_cosines (server.py:1913-1946) line-by-line.

      1: ∇‖r‖²    2: ∇‖J‖²_F (=∇ tr NTK)    3: ∇‖∇L‖²    4: ∇‖Q·Jᵀr‖²    5: ∇‖JJᵀr‖² (=∇‖NTK·r‖², two J's)    6: ∇‖f‖₂²    7: ∇‖ġ‖² (ġ=d(∇L)/dt=−H∇L)
    where r=Y−f, f=model output, Q=Σ_k Q_k (function Hessian), Jᵀr=Σ_k r_k ∇f_k, H=∇²L (loss Hessian). Returns (cosA[5], cosB[5], cosF[5], cosG[6], norms[7]):
    cosA=pairs [(1,2),(1,3),(1,4),(1,5),(2,3)], cosB=[(2,4),(2,5),(3,4),(3,5),(4,5)] (pairwise among 1-5);
    cosF=[cos(6,1),cos(6,2),cos(6,3),cos(6,4),cos(6,5)] (vector 6 vs the other five);
    cosG=[cos(7,1),cos(7,2),cos(7,3),cos(7,4),cos(7,5),cos(7,6)] (vector 7 vs the other six); norms=‖∇θφ_i‖ i=1..7."""
    import torch.func as _tf
    Yf = Y.reshape(-1)
    ff = lambda q: model.forward(q, X).reshape(-1)                      # f(θ) flat (M,)
    sumf = lambda q: ff(q).sum()
    Lval = lambda q: loss.value(model.forward(q, X), Y, N)

    def Jtr(q):                                                        # Jᵀr = Σ_k r_k ∇f_k  (r=Y−f at q)
        out, vjpf = _tf.vjp(ff, q)
        return vjpf(Yf - out)[0]

    def phi1(q):
        return ((Yf - ff(q)) ** 2).sum()                              # ‖r‖²

    def phi2(q):
        J = _tf.jacrev(ff)(q)
        return (J * J).sum()                                          # ‖J‖²_F = tr NTK

    def phi3(q):
        g = _tf.grad(Lval)(q)
        return (g * g).sum()                                          # ‖∇L‖²

    def phi4(q):                                                       # ‖(ΣQ_k)·Jᵀr‖²
        u = Jtr(q)
        Qu = _tf.jvp(lambda z: _tf.grad(sumf)(z), (q,), (u,))[1]
        return (Qu * Qu).sum()

    def phi5(q):                                                       # ‖J·Jᵀr‖² = ‖JJᵀr‖² = ‖NTK·r‖² (TWO J's)
        u = Jtr(q)
        Ju = _tf.jvp(ff, (q,), (u,))[1]
        return (Ju * Ju).sum()

    def phi6(q):                                                       # ‖f‖₂² (model output)
        f = ff(q)
        return (f * f).sum()

    def gL_fn(q):                                                       # ∇L (the gradient g)
        return _tf.grad(Lval)(q)

    def phi7(q):                                                        # ‖ġ‖²=‖H∇L‖² (ġ=d(∇L)/dt=−∇²L·∇L; sign drops in the norm)
        gL = gL_fn(q)
        HgL = _tf.jvp(gL_fn, (q,), (gL,))[1]
        return (HgL * HgL).sum()

    G = [_tf.grad(phi)(th) for phi in (phi1, phi2, phi3, phi4, phi5, phi6, phi7)]  # 7 vectors; v6=∇‖f‖², v7=∇‖ġ‖²

    def cs(i, j):
        ni = float(G[i].norm()); nj = float(G[j].norm())
        return float(G[i] @ G[j]) / (ni * nj) if (ni > 1e-30 and nj > 1e-30) else 0.0

    cosA = [cs(0, 1), cs(0, 2), cs(0, 3), cs(0, 4), cs(1, 2)]          # (1,2)(1,3)(1,4)(1,5)(2,3)
    cosB = [cs(1, 3), cs(1, 4), cs(2, 3), cs(2, 4), cs(3, 4)]          # (2,4)(2,5)(3,4)(3,5)(4,5)
    cosF = [cs(5, 0), cs(5, 1), cs(5, 2), cs(5, 3), cs(5, 4)]          # cos(∇‖f‖², vᵢ) for i=1..5  (§25 panel-2 plot 3)
    cosG = [cs(6, 0), cs(6, 1), cs(6, 2), cs(6, 3), cs(6, 4), cs(6, 5)]  # cos(∇‖ġ‖², vᵢ) for i=1..6  (§25 panel-2 plot 4)
    norms = [float(g.norm()) for g in G]                              # ‖∇θφ_i‖, i=1..7  (§25 panel-1 norms plot)
    return cosA, cosB, cosF, cosG, norms


def sec25_payload(model, loss, Jc, rr, th, X, Y, lr, N, outD, hist, t, is_mlp=None, rhist=None, r0hist=None):
    """§25 record. MIRRORS server.py's g25 block (server.py:3238-3265).

    Inputs (the per-step primitives diagnostics.compute() already has in scope):
      model, loss               — the eos_lab model + loss objects
      Jc (M_full, p), rr (M_full,) — per-output Jacobian (jac_cols) + generic residual (Y−f for MSE)
      th (p,), X, Y             — current parameters + data
      lr, N, outD              — GD step / sample count / output dim   (M = N·outD)
      hist                     — the rolling §25 history list (entries {"th","J","r","t"}, capped at 2);
                                 this function APPENDS the current tick (th,J,r,t) and pops to keep len ≤ 2.
      t                        — current step index (for the t−1,t−2 consecutivity check)
      is_mlp                   — whether to emit cosA/cosB/cosF/cosG/gnorms (MLP only). If None, inferred from
                                 getattr(model,"spec",None) is not None (the eos_lab MLP test, matching
                                 server's isinstance(model, MlpModel)).
      rhist                    — OPTIONAL rolling buffer (entries {"t","r"}, capped at 101) for cos(r_t, r_{t−k}),
                                 k∈{1,10,30,50,100}. If a list is passed, this function APPENDS (t,r) and emits cosR
                                 (all backends, not MLP-gated). If None, cosR is skipped. Mirrors server's sec25_rhist.

    Returns the g25 dict {gn,jdr,jrd,II,III,ddJr,cosJr (+cosA,cosB,cosF,cosG,gnorms if MLP; +cosR if rhist given)}.
    II/III are None until 3 consecutive ticks (t, t−1, t−2) are in `hist`. Side effects: mutates `hist` (and `rhist` if given)."""
    if is_mlp is None:
        is_mlp = getattr(model, "spec", None) is not None
    M = N * outD
    Jg25 = Jc[:M]                                                     # M×p effective-sample Jacobian
    rc25 = rr[:M].reshape(N, outD)                                    # residual as (N,outD) cotangent
    r25 = rr[:M]                                                      # flat residual (M,)
    gL25, _ = grad_loss(model, loss, th, X, Y)                        # ∇L (p,)
    JJg25 = Jg25.t() @ (Jg25 @ gL25)                                 # (JᵀJ)·∇L (p,)  [= J·ṙ for MSE]

    II25 = III25 = None
    if len(hist) >= 2 and hist[-1]["t"] == t - 1 and hist[-2]["t"] == t - 2:
        c225 = (lr / N) ** 2                                          # c² = (η/N)²
        h1, h2 = hist[-1], hist[-2]                                   # t−1, t−2
        g2_25 = h2["J"].t() @ h2["r"]                                 # g₂ = J_{t-2}ᵀ r_{t-2}  (p,)
        a25 = h1["r"] @ jac_hvp(model, h2["th"], X, g2_25)[:M]        # Q̃ g₂ = Σ_j r_{t-1,j} Q_{t-2,j} g₂  (p,)
        II25 = c225 * float((Jg25 * jac_hvp(model, h1["th"], X, a25)[:M]).sum())     # II = c²Σ_k ∇f_{t,k}ᵀ Q_{t-1,k}(Q̃ g₂)
        z25 = h1["J"].t() @ (h2["J"] @ g2_25)                        # z = J_{t-1}ᵀ J_{t-2} g₂  (p,)
        III25 = c225 * float((Jg25 * jac_hvp(model, h1["th"], X, z25)[:M]).sum())    # III = c²Σ_k ∇f_{t,k}ᵀ Q_{t-1,k} z

    Mr25 = hvp_S(model, th, X, rc25, gL25)                           # M_r∇L (p,);  J̇·r = −M_r∇L  (eos_lab signature: hvp_S(model,th,X,c,v))
    nJJ25 = float(JJg25.norm()); nMr25 = float(Mr25.norm())          # ‖J·ṙ‖, ‖J̇·r‖
    g25 = {"gn":  float(gL25.norm()),                                # 1. ‖∇L‖
           "jdr": nMr25,                                             # 2. ‖J̇·r‖ = ‖M_r ∇L‖
           "jrd": nJJ25,                                             # 3. ‖J·ṙ‖ = ‖JᵀJ ∇L‖
           "II":  II25,                                              # 4. §15 panel-1 term II (None until 3 consecutive steps)
           "III": III25,                                             # 5. §15 panel-1 term III
           "ddJr": float((JJg25 - Mr25).abs().sum()),               # plot 5: ‖J·ṙ + J̇·r‖₁ = ‖d/dt(J·r)‖₁
           "cosJr": ((-float(JJg25 @ Mr25)) / (nJJ25 * nMr25)) if (nJJ25 > 1e-30 and nMr25 > 1e-30) else 0.0}   # cos(J·ṙ, J̇·r)
    if is_mlp:                                                       # §25 panel-2 cosines: plots 2/3 (pairwise), plot 3 (cos ∇‖f‖² vs 5), plot 4 (cos ∇‖ġ‖² vs 6); panel-1 norms (7 ∇θ-vectors) — MLP, exact autograd
        g25["cosA"], g25["cosB"], g25["cosF"], g25["cosG"], g25["gnorms"] = _sec25_cosines(model, loss, th, X, Y, N, outD)

    if rhist is not None:                                            # §25 panel-1 plot 4: residual-direction drift — cos(r_t, r_{t−k}), k∈{1,10,30,50,100} (all backends)
        rc = r25.detach().clone(); nrc = float(rc.norm())
        rByT = {e["t"]: e["r"] for e in rhist}
        cosR = []
        for k in (1, 10, 30, 50, 100):                               # t−1 (odd) exposes the EoS step-to-step sign flip that the even lags alias
            rp = rByT.get(t - k); npk = None if rp is None else float(rp.norm())
            cosR.append((float(rc @ rp) / (nrc * npk)) if (rp is not None and nrc > 1e-30 and npk > 1e-30) else None)
        g25["cosR"] = cosR                                           # cos of r now vs r at t−{1,10,30,50,100} (None until that lag exists)
        if r0hist is not None:                                       # cos(r_t, r_0): cumulative rotation from the initial residual (bold reference line)
            if not r0hist: r0hist.append(rc.clone())                 # capture r_0 once
            r0 = r0hist[0]; n0 = float(r0.norm())
            g25["cosR0"] = (float(rc @ r0) / (nrc * n0)) if (nrc > 1e-30 and n0 > 1e-30) else None
        g25["rnorm"] = float(r25.reshape(N, outD).norm(dim=1).mean())   # ⟨‖r_i‖⟩_i: average over samples of the per-sample residual norm
        rhist.append({"t": t, "r": rc})
        if len(rhist) > 101:
            rhist.pop(0)

    hist.append({"th": th.detach().clone(), "J": Jg25.detach(), "r": r25.detach(), "t": t})
    if len(hist) > 2:
        hist.pop(0)
    return g25
