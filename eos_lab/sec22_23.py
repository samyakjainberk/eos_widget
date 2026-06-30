"""§22 + §23 — quadratic-Taylor surrogate REPLACE drivers (standalone), MIRRORS server._quad_surrogate_main /
server._sec15_surrogate / server._sec20_payload / server._qrandom_hvp / server._power_absmax.

§22 (s30, kind='frozen') and §23 (s31, kind='random') REPLACE the real GD loop: instead of stepping the actual
model, they iterate the closed-loop quadratic-Taylor surrogate from a freeze point θ_f and compute the CORE
Edge-of-Stability metrics ALONG it (so §1/§2/§3 + §20 + — if §15 on — the full §15 render the SURROGATE
trajectory rather than real GD). GD on the MSE of the surrogate function f̃ (η = lr/N); the per-sample function
Hessian Q̃_k is FROZEN at θ_f:
    f̃_k = f_k(θ_f) + J0_k·Δθ + ½ΔθᵀQ̃_kΔθ ,  J̃_k = J0_k + Q̃_k·Δθ ,  ∇²f̃_k = Q̃_k ,  Δθ = θ_sur − θ_f.
    ∇²L̃ = (1/N)J̃ᵀJ̃ + Σ_a(∂L/∂f̃_a)Q̃_a   (Gauss-Newton G + residual term S).
  kind='frozen' (§22):  Q̃_k = ∇²f_k(θ_f)  (the real model's per-sample function Hessian at the freeze point,
                        via jac_hvp / hvp_F / hvp_S at θ_f).
  kind='random' (§23):  Q̃_k = qhvp  — ONE shared low-rank random symmetric operator (the same for every k).
Because the 2nd-order Taylor is EXACT at θ_f, at st=0 every surrogate spectrum equals the real model's at θ_f.
Uses the SAME Lanczos seeds (0xA53F9 / 0x5EED1 / 0x6A17C / 0x7B23D) and the §20 cross-backend M_r start.

This module is exact-autograd-native (like all of eos_lab). The multi-sample primitives jac_cols / jac_hvp /
hvp_F / hvp_S mirror server's finite-difference definitions exactly, so the surrogate residual r̃, J̃, M_r
spectra and §15 terms match server to 1e-5..1e-13 in fp64; only the scalar-Hessian extreme spectra
(λmax(∇²L̃), function-Hessian / GN / S eigenvalues) carry the documented exact-vs-FD gap (≤~1e-6 fp64).
"""
import torch

from .models import (jac_cols, jac_hvp, hvp_F, hvp_S, grad_loss,
                     mlp_forward)
from .linalg import (safe_eigh, sym_eig_desc, lanczos_extreme_vals, lanczos_extreme,
                     randn_vec, sec15_stats, sec15_panel34, sec15_panelB)
from .sec16 import _randvec16


# §20 uses a fixed cross-backend mulberry32+gauss Lanczos start (NOT torch RNG) so the M_r eigenpairs match the
# server/browser bit-for-bit. SEC20_SEED == server.SEC20_SEED (0x205EE0). The §22/§23 surrogate-M_r histograms
# reuse the SAME start.
SEC20_SEED = 0x205EE0


# --------------------------------------------------------------------------- §23 random-Hessian primitives
def _qrandom_hvp(p, R, seed, dt, dev):
    """§23's fixed RANDOM function Hessian: a low-rank symmetric operator Q[v]=Σ_{j=1}^R s_j (ĝ_jᵀv) ĝ_j
    (ĝ_j unit-norm Gaussian, s_j=±1), UNSCALED. Seeded ⇒ deterministic. Returns the hvp closure.
    MIRRORS server._qrandom_hvp (torch.Generator RNG — IDENTICAL bytes given the same seed/dtype/device)."""
    gen = torch.Generator(device=dev); gen.manual_seed(int(seed) & 0x7FFFFFFF)
    g = torch.randn(R, p, generator=gen, dtype=dt, device=dev)
    g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-30)
    s = torch.randint(0, 2, (R,), generator=gen, device=dev).to(dt) * 2 - 1
    def hvp(v):
        return g.t() @ ((g @ v) * s)                       # Σ_j s_j (ĝ_jᵀv) ĝ_j
    return hvp


def _power_absmax(hvp, p, dt, dev, iters=24, seed=0xB17):
    """Top |eigenvalue| of a symmetric operator via power iteration (for scaling §23's random Q).
    MIRRORS server._power_absmax."""
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    v = torch.randn(p, generator=gen, dtype=dt, device=dev); v = v / v.norm().clamp_min(1e-30)
    lam = 0.0
    for _ in range(iters):
        w = hvp(v); nw = float(w.norm())
        if nw < 1e-30:
            break
        v = w / nw; lam = nw
    return lam


# --------------------------------------------------------------------------- §20 M_r payload (q0-Lanczos)
def _lanczos_q0(hvp, p, m, dt, dev, q0):
    """k≤m Lanczos steps (twice-iterated full reorthogonalisation) STARTED from a fixed vector q0 (the
    cross-backend mulberry32+gauss vector for §20). MIRRORS server._lanczos_core(..., q0=q0). eos_lab's
    linalg._lanczos_core has no q0 kwarg, so the §20 q0 start lives here. Returns (Q list, T (k×k fp64 CPU), k)."""
    q = q0.to(dtype=dt, device=dev)
    q = q / q.norm()
    Q, al, be = [], [], []
    qp = torch.zeros(p, dtype=dt, device=dev)
    beta = torch.zeros((), dtype=dt, device=dev)
    for _ in range(min(p, m)):
        Q.append(q)
        w = hvp(q)
        a = (w @ q)
        al.append(float(a))
        w = w - a * q - beta * qp
        Qm = torch.stack(Q)
        for _ in range(2):
            w = w - Qm.t() @ (Qm @ w)
        beta = w.norm()
        be.append(float(beta))
        if float(beta) < 1e-10:
            break
        qp = q
        q = w / beta
    k = len(Q)
    T = torch.zeros(k, k, dtype=torch.float64)
    for i in range(k):
        T[i, i] = al[i]
    for i in range(k - 1):
        T[i, i + 1] = be[i]; T[i + 1, i] = be[i]
    return Q, T, k


def sec20_payload(Jg, rt, th, X, N, outD, K, model=None, mr_hvp=None):
    """§20: spectral histograms of the RESIDUAL-WEIGHTED function Hessian M_r=Σ_k r_k Q_k and its alignment with
    the top Jacobian directions. From the top-K ⊕ bottom-K eigenpairs (λ_i, v_i) of M_r (matrix-free q0-Lanczos
    on hvpS(·,r) — or the surrogate M_r operator `mr_hvp` passed by §22/§23):
        lam      : {λ_i ÷N}                                (Plot 1 histogram)
        q1/q2/q3/q4 : {λ_i ÷N · |⟨v_i, u_k⟩|}  k=1..4      (Plots 2/3/4 histograms)  u_k = top-k right-singular vec of J.
    MIRRORS server._sec20_payload EXACTLY (same q0 start, top-K⊕bottom-K-distinct extraction, NTK-eigenvalue gate
    on u_k). For §22/§23 reuse, Jg/rt/th are the SURROGATE J̃/r̃/θ_f and mr_hvp the surrogate M_r operator."""
    M = N * outD
    p = Jg.shape[1]
    dt, dev = Jg.dtype, Jg.device
    rc = rt.reshape(N, outD)
    Klan = max(1, min(int(K), p))
    mlan = min(p, max(5 * Klan, 64))     # 5K: converge the top-K⊕bottom-K eigenpairs
    q0 = _randvec16(p, SEC20_SEED, dev, dt)                # mulberry32+gauss start (cross-backend identical)
    mhvp = mr_hvp if mr_hvp is not None else (lambda v: hvp_S(model, th, X, rc, v))   # default M_r=Σ_k r_k ∇²f_k at θ
    Q, T, k = _lanczos_q0(mhvp, p, mlan, dt, dev, q0)
    mu, Sv = safe_eigh(T)                                  # k Ritz pairs of M_r, ascending
    desc = torch.argsort(mu, descending=True)
    Qmat = torch.stack(Q)
    def ritz(ix):
        return Sv[:, ix].to(device=dev, dtype=dt) @ Qmat
    kt = min(Klan, k); bs = max(kt, k - Klan)             # top-K ⊕ bottom-K DISTINCT positions
    positions = list(range(kt)) + list(range(bs, k))
    Kc, Vc = sym_eig_desc(Jg @ Jg.t())                    # M×M NTK, eigvecs in columns (descending eigenvalue)
    ntol = 1e-9 * max(float(Kc[0]) if int(Kc.numel()) > 0 else 0.0, 1e-30)
    zero = torch.zeros(p, dtype=dt, device=dev)
    us = []
    for kk in range(4):                                   # top-4 right-singular vecs u_k = normalize(Jᵀ g_k); GATE on NTK eigenvalue
        if kk < M and float(Kc[kk]) > ntol:
            uk = Jg.t() @ Vc[:, kk]; nu = float(uk.norm())
            us.append(uk / nu if nu > 1e-30 else zero)
        else:
            us.append(zero)
    lam, q1, q2, q3, q4 = [], [], [], [], []
    scN = 1.0 / max(N, 1)
    for pos in positions:
        ix = int(desc[pos]); li = float(mu[ix]) * scN; vi = ritz(ix)
        lam.append(li)
        q1.append(li * abs(float(vi @ us[0])))
        q2.append(li * abs(float(vi @ us[1])))
        q3.append(li * abs(float(vi @ us[2])))
        q4.append(li * abs(float(vi @ us[3])))
    return {"lam": lam, "q1": q1, "q2": q2, "q3": q3, "q4": q4, "K": Klan}


# --------------------------------------------------------------------------- §15 along the surrogate
def _sec15_surrogate(model, loss, st, Jt, rt, hist, l_hvp, gss, th_freeze, X, N, outD, lr, p, kind, qhvp, divreg):
    """§15 along the §22/§23 surrogate trajectory. The surrogate's per-sample function Hessian Q̃_k is FROZEN
    (Q̇=0): kind='frozen' ⇒ Q̃_k=∇²f_k(θ_f); kind='random' ⇒ every Q̃_k is the shared random op qhvp. So the
    2nd-difference decomposition is EXACT and panels 3/4 (VII/VIII, the Q̇ terms) vanish. `hist` = the last ≤2
    emitted steps' {J,r,st}. Returns the available {g15,g15n,g15b,g15p34,g15corr} records.
    MIRRORS server._sec15_surrogate EXACTLY (eos_lab is exact-autograd-native, so the MLP per-sample frozen-Q
    path uses torch.func.jvp(grad(...)) exactly as the server does; reuses sec15_stats / sec15_panel34 / sec15_panelB)."""
    M = N * outD; dt, dev = Jt.dtype, Jt.device
    if kind == "frozen":
        qk_hvp = lambda v: jac_hvp(model, th_freeze, X, v)[:M]                 # {Q̃_k v}_k = {∇²f_k(θ_f) v}_k
        if getattr(model, "spec", None) is not None:                          # MLP: exact per-sample HVP (matches the FD jac_hvp path)
            import torch.func as _tf
            Xrep = X.repeat_interleave(outD, dim=0); OH = torch.eye(outD, dtype=dt, device=dev).repeat(N, 1)
            spec = model.spec
            def _f(tt, x, oh): return (mlp_forward(tt, spec, x.unsqueeze(0)).reshape(-1) * oh).sum()
            def persamp(v, k): return _tf.jvp(lambda q: _tf.grad(_f)(q, Xrep[k], OH[k]), (th_freeze,), (v,))[1]
        else:
            def persamp(v, k):
                ck = torch.zeros(M, dtype=dt, device=dev); ck[k] = 1.0
                return hvp_S(model, th_freeze, X, ck.reshape(N, outD), v)
    else:                                                                     # random: every Q̃_k = qhvp (shared)
        qk_hvp = lambda v: qhvp(v).unsqueeze(0).repeat(M, 1)
        persamp = lambda v, k: qhvp(v)
    out = {}
    consec = len(hist) >= 2 and hist[-1]["st"] == st - 1 and hist[-2]["st"] == st - 2
    jn = Jt.norm(dim=1); rn = rt.abs()                                        # panel 5 — per-sample norms
    acch = torch.zeros(M, dtype=dt, device=dev)
    for pr in range(4):
        Qv = qk_hvp(randn_vec(p, (0x5EC15 + pr * 0x9E3779B1) & 0xFFFFFFFF, dev, dt))
        acch = acch + (Qv * Qv).sum(dim=1)
    _ms = lambda x: [float(x.mean()), float(x.std(unbiased=False))]
    out["g15n"] = {"g": _ms(rn * jn), "j": _ms(jn), "h": _ms(torch.sqrt(acch / 4.0)), "r": _ms(rn)}
    mB = min(p, 48); TVl = []; TWl = []; BVl = []; BWl = []                   # panel B — per-sample top/bot eigvec projection
    AsumQJ = torch.zeros(p, dtype=dt, device=dev)
    for k in range(M):
        tv, bv, tw, bw = lanczos_extreme(lambda v, k=k: persamp(v, k), p, 1, mB, (0x5EC15B + k * 7919) & 0x7FFFFFFF, dev, dt)
        TVl.append(torch.stack(tv)); BVl.append(torch.stack(bv))
        TWl.append(torch.tensor(tw, dtype=dt, device=dev)); BWl.append(torch.tensor(bw, dtype=dt, device=dev))
        AsumQJ = AsumQJ + persamp(Jt[k], k)
    out["g15b"] = sec15_panelB(torch.stack(TVl)[:, 0], torch.stack(BVl)[:, 0],
                               torch.stack(TWl)[:, 0], torch.stack(BWl)[:, 0], Jt)
    Avec = 2.0 * AsumQJ; gL = Jt.t() @ gss                                    # panel 8 — A=∇tr(NTK)=2Σ Q̃_k J̃_k, B=2·∇²L̃·∇L̃
    gLn = float(gL.norm()); Bvec = 2.0 * l_hvp(gL)
    An = float(Avec.norm()); Bn = float(Bvec.norm()); inner = float(Avec @ Bvec)
    jnk = (Jt * Jt).sum(dim=1); jn2 = float(jnk.sum())
    Q1 = float(jnk.mean()); Q2 = float(((rt * rt) * jnk).mean()); dq1 = dq2 = D2corr = None
    if len(hist) >= 1 and hist[-1]["st"] == st - 1:
        Jp = hist[-1]["J"]; rp = hist[-1]["r"]; jn2p = (Jp * Jp).sum(dim=1)
        dq1 = Q1 - float(jn2p.mean()); dq2 = Q2 - float(((rp * rp) * jn2p).mean())
    if consec:
        D2corr = jn2 - 2.0 * float((hist[-1]["J"] ** 2).sum()) + float((hist[-2]["J"] ** 2).sum())
    out["g15corr"] = {"jn2": jn2, "An": An, "gn2": gLn * gLn, "Bn": Bn, "inner": inner,
                      "cos": (inner / (An * Bn) if (An > 1e-30 and Bn > 1e-30) else 0.0), "D2": D2corr,
                      "pred": 0.5 * lr * lr * inner, "q1": Q1, "q2": Q2, "dq1": dq1, "dq2": dq2}
    if consec:                                                                # panels 1-4 — 2nd-difference (3 consecutive steps); Q̇=0 ⇒ VII/VIII=0
        h1, h2 = hist[-1], hist[-2]
        sec, A15, B15, u1_15 = sec15_stats(qk_hvp, qk_hvp, Jt, h1["J"], h2["J"], h1["r"], h2["r"], lr, N, diveps=divreg)
        out["g15"] = sec
        out["g15p34"] = sec15_panel34(lambda thx, v: qk_hvp(v), th_freeze, Jt, h1["J"], h2["J"],
                                      h1["r"], h2["r"], u1_15, A15, B15, sec, lr, N, diveps=divreg)
    return out


# --------------------------------------------------------------------------- the surrogate driver
def quad_surrogate_main(model, loss, th_freeze, X, Y, N, outD, lr, steps, ee, K, n, mV, nResid,
                        s1, s2, s3, gs, kind, qhvp=None, s15=0, divreg=0.3):
    """§22/§23 REPLACE mode: iterate the closed-loop quadratic-Taylor surrogate from th_freeze and compute the
    CORE EoS metrics ALONG it — loss L̃, residual r̃, the loss-Hessian sharpness λmax(∇²L̃), and the
    function-Hessian / Gauss-Newton(=NTK) / residual-weighted extreme spectra — plus the §20 M_r payload, plus
    (if s15) the full §15 panels on the surrogate. GD on the MSE of f̃ (η = lr/N); Q̃ frozen at θ_f.
    kind='frozen' (§22): Q̃_k=∇²f_k(θ_f); kind='random' (§23): Q̃_k=qhvp (one shared op).
    Yields (surrogate_step, step_fields, sec20_payload, sec15r) every ee steps. MSE only.
    MIRRORS server._quad_surrogate_main EXACTLY (signature 1:1 + the `model`/`loss` made explicit since eos_lab
    is not module-global like server's _TL). Seeds 0xA53F9 / 0x5EED1 / 0x6A17C / 0x7B23D as the real §1-§3 block."""
    dt, dev = th_freeze.dtype, th_freeze.device
    J0, f0flat = jac_cols(model, th_freeze, X)            # (M,p), (M,) frozen at the expansion point
    Yf = Y.reshape(-1)
    p = J0.shape[1]; M = N * outD
    needVals = s1 or s2 or s3
    # the summed function Hessian Σ_a Q̃_a is FROZEN at θ_f, so its extreme eigenvalues are computed ONCE.
    feTop = feBot = None
    if needVals:
        hvpF_sur = (lambda v: hvp_F(model, th_freeze, X, v)) if kind == "frozen" else (lambda v: M * qhvp(v))
        feTop, feBot = lanczos_extreme_vals(hvpF_sur, p, n, mV, 0xA53F9, dev, dt)
    th_sur = th_freeze.clone()
    sur15_hist = []                                       # §15-on-surrogate: last ≤2 emitted steps' {J,r,st}
    for st in range(int(steps) + 1):
        dth = th_sur - th_freeze
        if kind == "frozen":
            QD = jac_hvp(model, th_freeze, X, dth)        # (M,p): ∇²f_k(θ_f)·Δθ per output
            fss = f0flat + (J0 @ dth) + 0.5 * (QD @ dth)
            Jqs = J0 + QD
        else:                                             # random: single shared Q
            qd = qhvp(dth)                                # (p,)
            fss = f0flat + (J0 @ dth) + 0.5 * float(dth @ qd)
            Jqs = J0 + qd.unsqueeze(0)                    # broadcast J0_k + qd
        rt = Yf - fss                                     # surrogate residual r̃ = Y − f̃
        if not torch.isfinite(fss).all() or float(rt @ rt) > 1e14:
            break                                         # diverged → keep the snapshots so far
        if st % ee == 0:
            cS = loss.resid_cotangent(fss.reshape(N, outD), Y, N)        # (N,outD) loss cotangent ∂L/∂f̃
            lossv = float(loss.value(fss.reshape(N, outD), Y, N))
            if kind == "frozen":
                s_hvp = lambda v: hvp_S(model, th_freeze, X, cS, v)      # S-term = Σ_a (∂L/∂f̃_a) Q̃_a v
                sec = sec20_payload(Jqs, rt, th_freeze, X, N, outD, K, model=model)   # M_r·v defaults to hvpS(θ_f,·,r̃) inside
            else:
                mr_hvp = (lambda v, rs=float(rt.sum()): rs * qhvp(v))
                s_hvp = (lambda v, cs=float(cS.sum()): cs * qhvp(v))
                sec = sec20_payload(Jqs, rt, th_freeze, X, N, outD, K, model=model, mr_hvp=mr_hvp)
            g_hvp = lambda v: (Jqs.t() @ (Jqs @ v)) / N                  # Gauss-Newton (1/N)J̃ᵀJ̃ v (= NTK)
            l_hvp = lambda v: g_hvp(v) + s_hvp(v)                        # loss-Hessian ∇²L̃ = G + S
            hlt, hlb = (lanczos_extreme_vals(l_hvp, p, n, mV, 0x5EED1, dev, dt) if needVals else (None, None))
            gnt = gnb = srt = srb = None
            if (s2 or s3) and gs:
                gnt, gnb = lanczos_extreme_vals(g_hvp, p, n, mV, 0x6A17C, dev, dt)
                srt, srb = lanczos_extreme_vals(s_hvp, p, n, mV, 0x7B23D, dev, dt)
            step_fields = {
                "loss": lossv, "sharp": (hlt[0] if hlt else None),
                "r": rt[:nResid].detach().cpu().tolist(),
                "hfMax": (feTop[0] / N if feTop else None), "hfMin": (feBot[0] / N if feBot else None),
                "hfTop": ([v / N for v in feTop[:n]] if feTop else None),
                "hfBot": ([v / N for v in feBot[:n]] if feBot else None),
                "hlTop": hlt, "hlBot": hlb, "gnTop": gnt, "gnBot": gnb, "srTop": srt, "srBot": srb,
            }
            sec15r = {}
            if s15:                                                     # §15 on the surrogate trajectory (frozen Q̃)
                sec15r = _sec15_surrogate(model, loss, st, Jqs, rt, sur15_hist, l_hvp, cS.reshape(-1),
                                          th_freeze, X, N, outD, lr, p, kind, qhvp, divreg)
                sur15_hist.append({"J": Jqs.detach().clone(), "r": rt.detach().clone(), "st": st})
                if len(sur15_hist) > 2:
                    sur15_hist.pop(0)
            yield (st, step_fields, sec, sec15r)
        gss = loss.resid_cotangent(fss.reshape(N, outD), Y, N).reshape(-1)   # MSE: (f̃−Y)/N = −r̃/N
        th_sur = th_sur - lr * (Jqs.t() @ gss)


def quad_surrogate_driver(model, loss, th0, X, Y, N, outD, lr, steps, ee, K, n, mV, nResid,
                          s1, s2, s3, gs, opt="gd", s15=0, divreg=0.3,
                          s30=0, s30t=0, s31=0, s31r=200, s31scale=1.0, seed=0):
    """Top-level §22/§23 entry point (the eos_lab analog of server's run_stream `do_quad_sur` branch). Selects
    frozen-Q (§22, s30) vs random-Q (§23, s31; frozen wins if both on), builds the freeze point / random
    operator, runs quad_surrogate_main, and yields TYPED records mirroring server's emitted stream:
        {"type":"step","t":st,"surrogate":rectype, **step_fields}          (the §1-§3 surrogate step record)
        {"type":rectype,"t":st, **sec20}                                   (the §22/§23 section's own M_r slider)
        {"type":<g15 key>,"t":st, **rec}  for each §15-on-surrogate panel  (only if s15)
    rectype = 'g22' (frozen) or 'g23' (random). MIRRORS server run_stream lines 3088-3119.
    NB: server ALSO emits a duplicate {"type":"g20",...} when s28 is on (§20 section ↔ surrogate M_r); that is a
    pure re-tag of the SAME sec20 dict and is wired at integration time, so it is omitted from this standalone
    driver (the authoritative M_r data is in the rectype record)."""
    dt, dev = th0.dtype, th0.device
    p = model.p
    Xq, Yq = X, Y
    if s30:                                                              # §22 frozen-Q (priority if both on)
        thf = th0.detach().clone()
        for _ in range(int(s30t)):                                       # advance the REAL model to iteration t (θ_t)
            from .models import opt_dir
            thf = thf - lr * opt_dir(model, grad_loss(model, loss, thf, Xq, Yq)[0], opt)
        gen = quad_surrogate_main(model, loss, thf, Xq, Yq, N, outD, lr, steps, ee, K, n, mV, nResid,
                                  s1, s2, s3, gs, "frozen", s15=s15, divreg=divreg)
        rectype = "g22"
    else:                                                                # §23 random-Q (from θ₀)
        qh = _qrandom_hvp(p, int(s31r), (int(seed) * 2654435761 + 0x235EE0) & 0x7FFFFFFF, dt, dev)
        tgt = _power_absmax(lambda v: hvp_F(model, th0, Xq, v), p, dt, dev) * float(s31scale)  # scale random Q to real fn-Hessian λmax at θ₀
        traw = _power_absmax(qh, p, dt, dev)
        sc = (tgt / traw) if traw > 1e-30 else 1.0
        gen = quad_surrogate_main(model, loss, th0.detach().clone(), Xq, Yq, N, outD, lr, steps, ee, K, n, mV, nResid,
                                  s1, s2, s3, gs, "random", qhvp=(lambda v, c=sc: c * qh(v)), s15=s15, divreg=divreg)
        rectype = "g23"
    for (st, step_fields, sec20, sec15r) in gen:
        yield {"type": "step", "t": st, "steps": steps, "p": p, "surrogate": rectype, **step_fields}
        yield {"type": rectype, "t": st, **sec20}
        for _t15, _r15 in sec15r.items():
            yield {"type": _t15, "t": st, **_r15}
