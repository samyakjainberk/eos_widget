"""
prediction — the EoS "prediction widget" forecasts (server.py's `/prediction` route), ported to eos_lab.

MIRRORS:  server.py  (_first_signchange / _max_abs_curvature / _pred_signchange / run_sweep [incl. the
                       EoS-guard lr rescue, the magnified-PS `ps5x`, and the dedicated 2c/2d σ-sweep], and
                       the per-step forecast blocks g_pred3 / g_pred3m / g_pred4 / g_pred4m / g_trace /
                       g_pred6 / g_ray inlined in run_stream's main loop)
          ↔ index_prediction.html  (the 4-plot forecast panel, the ★ Prediction-4.2 ray panel, + the §1/§2 sweep scatter)

The prediction widget layers three families of theoretical forecasts on top of the ACTUAL GD trajectory:

  Section 1 & 2 — a SWEEP over (lr, init-std) pairs (`sweep`). Each pair runs a light full-batch GD and
    records where d = ‖J·ṙ‖ − ‖J̇·r‖ (= jrd − jdr, §25's product-rule split) first changes sign — both the
    ACTUAL sign-change (`tAct`) and two PREDICTED from θ(Tstart) via the §10 Eq-51 self-trajectory (cubic `tPred`
    and quadratic `tPredQ`) with the model refrozen every K steps. Plus the loss curvature (`curv`), the
    `swsumn`-window sums (`sumD`/`sumCurv`/`sumD2`/`sumJgrad`), and the top-3-NTK residual alignment (`alignNTK`).

  Prediction-3 (`Pred3Tracker`) — after d first changes sign at t*, freeze J*=J(t*) and compare the ACTUAL
    residual with a frozen-J linear NTK theory r_{k+1}=(I−(η/N)J*J*ᵀ)r_k (`rThy`/`cos`) AND a 2nd-order theory
    that adds the ½(η/N)²(gᵀQ*g) curvature term (`rThy2`/`cos2`). The multi-freeze panel repeats this with (J,Q)
    frozen at t*−10 / t*+10 / t*+20 (with a back-fill of the earliest anchor).

  Prediction-4 (`Pred4Tracker`) — at t0 freeze θ0, J0=J(t0), the residual r0, and propagate the Jacobian
    J_{s+1}=(I+(η/N)M_r)J_s with M_r=Σ_k r0_k Q_k(θ0) (matrix-free via hvp_S), reporting the top-3 NTK eigvals
    λ(JJᵀ) of the predicted vs actual J (`kPred`/`kAct`) and NTK/GN eigvector cosines. A second variant lets the
    residual EVOLVE (`kPredE`, re-anchored to the actual residual every s steps). The multi-freeze panel repeats
    at t0−10 / t0+10 / t0+20.

  Prediction-5 / trace (`TraceTracker`) — at stat_init start four frozen-Q Jacobian trajectories
    {quad(T=0), cubic}×{live, self} that RE-ANCHOR at the actual state every window, and report the predicted
    Tr(NTK)=‖J‖²_F/N (with and without the per-step PSD term) vs the actual Tr(JᵀJ)/N (`trGN`) and
    Tr(∇²L)/N (`trHess`, via 24 Hutchinson probes of the EXACT loss-Hessian hvp_L).

  Prediction-6 (`Pred6Tracker`) — four independent optimizer trajectories {GD, Signed-GD, Muon, Gauss-Newton}
    from a shared θ₀; per tick the top-50 Gauss-Newton scree, loss, and Lanczos sharpness of each.

  Prediction-4.2 / Phase-2a ray (`RayTracker`, GD-only) — MULTI-ANCHOR: each time cos(∇L, w₁) rises up through
    `prthr` in a direction different from the previous anchor (w₁ = top eigvec of M_r=Σ_k r_kQ_k, warm-started
    Lanczos), freeze θ_a/r_a/w₁ and walk the straight ray θ_a+s·w₁, predicting σ₁(J)/‖J‖_F, the step-clock, and
    the D_J=D_r phase-2a exit; the span is sized by a `prK`-step GD lookahead. `step()` → the g_ray dict.

PARITY NOTE (FD-vs-exact).  server.py's MlpModel uses FINITE-DIFFERENCE HVPs (EPS=1e-3) for hvp_S / hvp_L,
whereas eos_lab uses EXACT double-backward autograd. jac_hvp is FD on BOTH backends (EPS=1e-3), so every
jac_hvp-derived quantity (prediction-3's quad2, prediction-5's Q0[g]/T[a,b]) matches server to ~1e-6 in fp64.
Quantities that go through hvp_S (prediction-3's jdr, prediction-4's M_r·J propagation) or hvp_L
(prediction-5's trHess) carry the documented FD-vs-exact gap (~1e-3 relative in fp64) — NOT a port bug.
The pure-linear pieces (jrd = ‖JᵀJ∇L‖, the (I−(η/N)JJᵀ) residual recursions, NTK eigenvalues of the
frozen-residual Jacobian) match to ~1e-10.

The eos_lab primitives used (all from eos_lab.models):
  grad_loss(model, loss, th, X, Y) -> (∇L, out)        [server gradL]
  jac_cols(model, th, X)           -> (J (M,p), out)    [server jac_cols]
  jac_hvp(model, th, X, z)         -> {∇²f_a·z}_a (M,p)  central-diff, EPS=1e-3  [server jac_hvp]
  hvp_S(model, th, X, c, v)        -> (Σ_a c_a ∇²f_a)·v   NOTE arg order (c, v)  [server hvpS(th,X,v,c)]
  hvp_L(model, loss, th, X, Y, v)  -> (∇²L)·v            [server hvpL]
and eos_lab.linalg.{safe_eigvalsh (ascending, = server._safe_eigvalsh), randn_vec (= server._randn_vec)}.
"""
import math

import torch

from .models import (build_model, build_loss, grad_loss, jac_cols, jac_hvp,
                     hvp_S, hvp_L, opt_dir, precond_flow, MlpModel)
from .data import init_data_theta
from .linalg import safe_eigvalsh, safe_eigh, randn_vec, lanczos_extreme_vals, _lanczos_core
from .sec16 import _randvec16   # mulberry32/gauss Lanczos start vector — IDENTICAL across server/eos_lab/browser (matches server._randvec16), so warm/cold Lanczos starts are byte-faithful

SEC21_SEED = 0x205EE0   # Pred-4.2 ray cold-start Lanczos seed (server SEC21_SEED = SEC20_SEED = 0x205EE0)


# ═══════════════════════════════════════════════════════════════════════ pure-Python helpers (copied)
def first_signchange(diffs, cap):
    """First index where the sign of `diffs` flips (skipping exact 0s). Returns (t, censored):
    (t, False) if it flips within the run, else (cap, True). MIRRORS server._first_signchange (copied)."""
    prev = None
    for i, d in enumerate(diffs):
        s = 1 if d > 0 else (-1 if d < 0 else 0)
        if s == 0:
            continue
        if prev is not None and s != prev:
            return i, False
        prev = s
    return cap, True


def max_abs_curvature(ys):
    """Prediction-2 loss curvature: the discrete central 2nd difference L[t+1]−2L[t]+L[t−1] (unit time step)
    of largest ABSOLUTE magnitude over the loss series, keeping its sign (peak signed d²loss/dt²).
    MIRRORS server._max_abs_curvature (copied)."""
    n = len(ys)
    if n < 3:
        return 0.0
    best = 0.0
    for t in range(1, n - 1):
        d2 = float(ys[t + 1]) - 2.0 * float(ys[t]) + float(ys[t - 1])
        if abs(d2) > abs(best):
            best = d2
    return float(best)


def curvature_at(ys, t):
    """Prediction-2 (iteration-t variant): the discrete central 2nd difference L[t+1]−2L[t]+L[t−1] of the loss
    AT a specific iteration t (its exact sign), NOT the peak. 0.0 if t is outside the finite range [1, len−2].
    MIRRORS server._curvature_at (copied)."""
    n = len(ys)
    if n < 3 or t < 1 or t > n - 2:
        return 0.0
    return float(ys[t + 1]) - 2.0 * float(ys[t]) + float(ys[t - 1])


# ═══════════════════════════════════════════════════════════════════════ predicted sign-change (§10 cubic)
def pred_signchange(model, loss, th_start, X, Y, N, outD, p, lr, K, max_steps, cubic=True, opt="gd"):
    """PREDICTED first sign-change of d = ‖J·ṙ‖−‖J̇·r‖ along the Eq-51 (§10) trajectory from th_start,
    with the quadratic model refrozen every K steps. Returns (t, censored). The residual is the frozen-quad
    self-residual r_q; jrd/jdr use the propagated J and r_q with Q frozen at the current anchor. Sign
    only ⇒ ∇L magnitude is irrelevant. MIRRORS server._pred_signchange exactly (matrix-free via eos_lab).
    cubic=True  ⇒ full §10 cubic: J += (η/N)·Qg + ½(η/N)²·T[g,g], Qg carries the 3rd-derivative T corrections.
    cubic=False ⇒ QUADRATIC approximation (prediction-1 plot 1): T=0 and Q FIXED ⇒ J += (η/N)·Q₀·g only."""
    dev, dt = th_start.device, th_start.dtype
    M = N * outD
    eps = 1e-2
    Yf = Y.reshape(-1)[:M]

    def T2m(th0, a, b):                                   # multi 3rd-derivative T[a,b] via central diff of Q[b]
        an = float(a.norm())
        if an < 1e-30:
            return torch.zeros(M, p, dtype=dt, device=dev)
        ah = a / an
        return an * (jac_hvp(model, th0 + eps * ah, X, b)[:M]
                     - jac_hvp(model, th0 - eps * ah, X, b)[:M]) / (2 * eps)

    def anchor(thc):                                     # (re)freeze the quadratic model at thc
        J0, _ = jac_cols(model, thc, X)
        f0 = model.forward(thc, X).reshape(-1)[:M]
        return {"th0": thc.clone(), "J0": J0[:M].clone(), "f0": f0.clone(),
                "J": J0[:M].clone(), "Dth": torch.zeros(p, dtype=dt, device=dev), "HistG": []}

    st = anchor(th_start)
    prev = None
    for s in range(max_steps):
        if s > 0 and s % K == 0:                          # refreeze every K steps at the current predicted θ
            st = anchor(st["th0"] + st["Dth"])
        th0 = st["th0"]; Jc = st["J"]; dth = st["Dth"]
        HzD = jac_hvp(model, th0, X, dth)[:M]
        r_q = Yf - (st["f0"] + st["J0"] @ dth + 0.5 * (HzD @ dth))    # frozen-quad self-residual
        g, scale = precond_flow(model, opt, Jc, r_q, N, lr)   # (direction, step) recomputed each step; gd ⇒ (Jᵀr_q, η/N)
        jrd = float((Jc.t() @ (Jc @ g)).norm())           # ‖J·ṙ‖ ∝ ‖JᵀJ·g‖ (only sign(jrd−jdr) matters for the sign-change)
        jdr = float(hvp_S(model, th0, X, r_q.reshape(N, outD), g).norm())   # ‖J̇·r‖ ∝ ‖M_r·g‖ (Q frozen at th0)
        d = jrd - jdr
        sg = (1 if d > 0 else (-1 if d < 0 else 0))
        if prev is not None and sg != 0 and sg != prev:
            return s, False
        if sg != 0:
            prev = sg
        Qg = jac_hvp(model, th0, X, g)[:M]                # advance the self trajectory (Eq-51)
        if cubic:
            for gk in st["HistG"]:
                Qg = Qg + scale * T2m(th0, gk, g)
            dJ = scale * Qg + 0.5 * (scale ** 2) * T2m(th0, g, g)   # full cubic (3rd-derivative T corrections)
            st["HistG"].append(g)
        else:
            dJ = scale * Qg                               # QUADRATIC: T=0, Q fixed
        st["J"] = Jc + dJ
        st["Dth"] = dth + scale * g
    return max_steps, True


# ═══════════════════════════════════════════════════════════════════════ Section 1 & 2 — (lr, std) sweep
def sweep(cfg, npairs=None, K=None, Tstart=None, steps=None, seed=None,
          swsumn=None, swsumn2=None, lr_range=None, std_range=None, device=None, dtype=None, cifar_dir=None):
    """Prediction-widget Sections 1 & 2 — sweep `npairs` random (lr, init-std) pairs (log-uniform in the
    ranges). Per pair a light full-batch GD run tracks d = ‖J·ṙ‖−‖J̇·r‖ = jrd−jdr (§25) + the loss, yielding
    one dict per pair with keys:
      lr, std          — the sampled pair
      tAct, censA      — first step d changes sign (censored at `steps` if none; censA true if censored OR diverged)
      tPred, censP     — the same sign-change PREDICTED from θ(Tstart) via §10 CUBIC Eq-51, refrozen every K
      tPredQ, censPQ   — the QUADRATIC (T=0, Q fixed) prediction (prediction-1 plot 1)
      alignNTK         — mean over the first `swsumn` steps of Σ_{k≤3}|⟨r̂, v_k⟩| (residual vs top-3 NTK eigvecs; tick colour)
      sumD, sumCurv    — prediction-2: Σ d and Σ d²loss/dt² over the first `swsumn` iterations (from 0)
      sumD2, sumJgrad  — prediction-2b: Σ d and Σ Δ‖J‖_F (= ‖J‖_{swsumn2} − ‖J‖_0, telescoping) over `swsumn2` iters
      jfro             — windowed-mean ‖J‖_F over `swsumn` steps (prediction-2/2b RIGHT-half tick colour)
      lr, lrOrig       — the EFFECTIVE lr (after any EoS rescue) and the pair's ORIGINAL swept lr
      sharp0, eosDrop, rescued — EoS guard: λmax(∇²L) at init; started at EoS & un-rescuable ⇒ eosDrop (drop from plots
                          3-4); rescued ⇒ lr was lowered (lr_eff) so the pair starts below 2/η
      ps5x, ps5t       — magnified-PS: Σ_{k<swps5n} u₁ᵀ(QJr_k)v₁ with Q,J frozen at the PS-ONSET step ps5t and r evolving
                          (t=0 if the pair starts in PS, else the neg→pos jrd−jdr crossing — matching Pred-3); None ⇒
                          the pair never enters PS below EoS. The magnified-PS y-axis is sumJgrad.
      curv             — peak signed d²loss/dt² (max-|·| central 2nd difference of loss(t))
      swsumn, swsumn2  — the two summation windows echoed back
      diverged         — the pair blew up (loss NaN or >1e12) before `steps`

    When `cfg.sw2cd` or `cfg.sw2e`, ALSO yields (FIRST, before the pairs) the dedicated 2c/2d/2e σ-sweep —
    `{type:'sweeppt2c', i, std, xdiff, xnorm, x2e, dj, djn, l, swmrk, n2c}` — per init scale σ (log-spaced over
    [swcsmin, swcsmax]): the curvature-sign-split gradient alignment A±=Σ_{σ_i≷0} σ_i·(u_iᵀg)²/‖g‖² (g=Jᵀr;
    (σ_i,u_i)=top-k⊕bottom-k eigenpairs of M_r=Σ_k r_kQ_k; SQUARED projection ⇒ gauge-invariant) → 2c x=`xdiff`=A₊−A₋,
    2d x=`xnorm`=(A₊−A₋)/(A₊+A₋); and the full-operator Rayleigh 2e x=`x2e`=gᵀM_r g (via hvp_S); all paired with the
    short-horizon Δ‖J‖_F over l=`swjl` GD steps → 2c/2e y=`dj`=‖J_l‖−‖J₀‖, 2d y=`djn`=(‖J_l‖−‖J₀‖)/(‖J₀‖+ε).
    Pair dicts carry `type:'sweeppt'`.

    A generator (yields as the pairs are computed, like server.run_sweep's SSE stream). MIRRORS
    server.run_sweep (the trajectory / diagnostics run through eos_lab primitives instead of the globals).

    Config knobs (fall back to the cfg attribute / the server default when the arg is None):
      npairs ← cfg.swpairs (100)     K ← cfg.swK (20)        Tstart ← cfg.swTstart (0)
      swsumn ← cfg.swsumn (20)       swsumn2 ← cfg.swsumn2 (20)
      steps  ← cfg.steps             seed ← cfg.seed
      lr_range ← (cfg.swlrmin, cfg.swlrmax) or auto base_lr/4 … base_lr·4
      std_range ← (cfg.swstdmin, cfg.swstdmax) or auto base_std/4 … base_std·4
    """
    from .train import resolve_device, effective_dims
    dev = resolve_device(device)
    if dtype is None:
        dtype = torch.float32 if dev.type == "cuda" else torch.float64
    torch.set_grad_enabled(False)

    P = cfg.P()
    dataset = cfg.dataset
    inD, outD = effective_dims(cfg)
    model = build_model(cfg.arch, inD, outD, P, dev, dtype)
    loss = build_loss(cfg.loss)
    if loss.name != "mse":
        raise ValueError("sweep needs MSE loss (the ‖J·ṙ‖−‖J̇·r‖ theory is for squared loss).")
    p = model.p
    N = int(cfg.nsamp); M = N * outD
    if M > 400 or M * p > 200_000_000:
        raise ValueError(f"sweep too large (M={M}, p={p}); use a small MLP + small nsamp.")

    steps = max(5, int(cfg.steps if steps is None else steps))
    npairs = max(1, min(200, int(getattr(cfg, "swpairs", 100) if npairs is None else npairs)))
    K = max(1, int(getattr(cfg, "swK", 20) if K is None else K))
    Tstart = max(0, min(steps - 2, int(getattr(cfg, "swTstart", 0) if Tstart is None else Tstart)))
    swsumn = max(1, int(getattr(cfg, "swsumn", 20) if swsumn is None else swsumn))     # prediction-2 window (Σ d, Σ d²loss/dt²)
    swsumn2 = max(1, int(getattr(cfg, "swsumn2", 20) if swsumn2 is None else swsumn2))   # prediction-2b window (Σ d, Σ Δ‖J‖)
    seed0 = int(cfg.seed if seed is None else seed)
    base_lr = float(cfg.lr); base_std = float(cfg.init)
    opt = getattr(cfg, "optimizer", "gd")   # sweep actual + predicted trajectories follow the chosen optimizer

    if lr_range is not None:
        lrmin, lrmax = float(lr_range[0]), float(lr_range[1])
    else:
        _lrmn = float(getattr(cfg, "swlrmin", 0) or 0); _lrmx = float(getattr(cfg, "swlrmax", 0) or 0)
        lrmin = _lrmn if _lrmn > 0 else base_lr / 4; lrmax = _lrmx if _lrmx > 0 else base_lr * 4
    if std_range is not None:
        stdmin, stdmax = float(std_range[0]), float(std_range[1])
    else:
        _stmn = float(getattr(cfg, "swstdmin", 0) or 0); _stmx = float(getattr(cfg, "swstdmax", 0) or 0)
        stdmin = _stmn if _stmn > 0 else base_std / 4; stdmax = _stmx if _stmx > 0 else base_std * 4
    lrmin = max(1e-6, lrmin); lrmax = max(lrmin * 1.01, lrmax)
    stdmin = max(1e-6, stdmin); stdmax = max(stdmin * 1.01, stdmax)

    _, X, Y, _, _ = init_data_theta(model, P, dataset, N, inD, outD, dev, dtype, cifar_dir)

    import random as _random
    rng = _random.Random(seed0 or 1)                       # deterministic log-uniform (lr, std) pairs (server parity)

    swps5n = max(1, int(getattr(cfg, "swps5n", 4)))        # magnified-PS: #frozen-Q,J / evolving-r steps to sum u₁ᵀ(QJr)v₁ over
    _EOS_SEED = 0x5A1DBEEF & 0x7FFFFFFF                     # server _lanczos_extremes start seed (fed through _randvec16 = mulberry32/gauss, NOT torch.randn, so the truncated Krylov subspace matches the server bit-for-bit)
    def _sharp_top(hvp):                                    # λmax via matrix-free Lanczos (server _lanczos_extremes top eigenvalue)
        try:
            _Q, _T, _kl = _lanczos_core(hvp, p, min(p, 28), 0, dev, dtype, q0=_randvec16(p, _EOS_SEED, dev, dtype))
            _mu, _ = safe_eigh(_T)
            return float(_mu[-1])
        except Exception:
            return None
    def _eds_pairs(hvp, K):                                # top-K ⊕ bottom-K eigenpairs (λ, unit vec) of `hvp`'s operator
        # mirrors server._eds_eigpairs: SEC21_SEED == server SEC20_SEED == 0x205EE0; mlan = max(5K, 64) resolves both ends.
        Klan = max(1, min(int(K), p)); mlan = min(p, max(5 * Klan, 64))
        _Q, _T, _k = _lanczos_core(hvp, p, mlan, 0, dev, dtype, q0=_randvec16(p, SEC21_SEED, dev, dtype))
        _mu, _Sv = safe_eigh(_T); _desc = torch.argsort(_mu, descending=True); _Qmat = torch.stack(_Q)
        def _ritz(ix):
            return _Sv[:, ix].to(device=dev, dtype=dtype) @ _Qmat
        kt = min(Klan, _k); bs = max(kt, _k - Klan)        # top-K ⊕ bottom-K DISTINCT positions (no double-count when 2K>k)
        positions = list(range(kt)) + list(range(bs, _k))
        return [(float(_mu[int(_desc[pos])]), _ritz(int(_desc[pos]))) for pos in positions]

    # ── Prediction-2c/2d/2e (dedicated σ-sweep): σ log-spaced over [swcsmin, swcsmax] (default 0.1→10), decoupled from
    #    the (lr,σ) pairs below. Per init scale σ, at t=0 measure how the residual-contracted gradient g=Jᵀr aligns
    #    with the POSITIVE- vs NEGATIVE-curvature of M_r=Σ_k r_kQ_k, paired against the short-horizon change in ‖J‖_F
    #    over l GD steps. A± = Σ_{σ_i≷0} σ_i·(u_iᵀg)² / ‖g‖² (projection SQUARED ⇒ GAUGE-INVARIANT; A₊−A₋ = the signed
    #    Rayleigh gᵀM_r g/‖g‖² over the ±k eigenspace). 2c: x=A₊−A₋, y=‖J_l‖−‖J₀‖. 2d: x=(A₊−A₋)/(A₊+A₋), y=(‖J_l‖−‖J₀‖)/(‖J₀‖+ε).
    #    2e: x=gᵀM_r g (FULL operator via one hvp_S, NOT ±k-restricted), y=‖J_l‖−‖J₀‖. MIRRORS server.run_sweep (~6007-6062).
    _sw2cd = bool(int(getattr(cfg, "sw2cd", 0)))           # OFF by default: 2c/2d (plots 4-5)
    _sw2e = bool(int(getattr(cfg, "sw2e", 0)))             # OFF by default: 2e (plot 6), independent toggle
    swmrk = max(1, int(getattr(cfg, "swmrk", 10)))
    swcsmin = max(1e-6, float(getattr(cfg, "swcsmin", 0.1) or 0.1))
    swcsmax = max(swcsmin * 1.01, float(getattr(cfg, "swcsmax", 10.0) or 10.0))
    _lgd = float(getattr(cfg, "lr", 0.05)); _ljl = max(1, int(getattr(cfg, "swjl", 4))); _EPS2D = 1e-12   # GD lr, lookahead l (k0=0), 2d ε
    _lo, _hi = math.log(swcsmin), math.log(swcsmax)
    for si in range(npairs if (_sw2cd or _sw2e) else 0):   # range(0) ⇒ skip entirely when BOTH toggles are off
        std_c = math.exp(_lo + (_hi - _lo) * (si / max(1, npairs - 1)))   # deterministic log-spaced σ grid
        th_c = model.init_theta(seed0 + 1, std_c)
        out_c = model.forward(th_c, X)
        if not (float(loss.value(out_c, Y, N)) < 1e12):
            continue                                       # skip a NaN/blow-up init (rare at large σ)
        rr_c = (-N * loss.resid_cotangent(out_c.reshape(N, outD), Y, N)).reshape(-1)[:M]   # r = y − f
        Jc0, _ = jac_cols(model, th_c, X); Jm0 = Jc0[:M]; j0 = float(Jm0.norm())            # J at init (M×p); ‖J₀‖_F
        g = Jm0.t() @ rr_c; gn = max(float(g.norm()), 1e-30); g2 = gn * gn                  # g = Jᵀr; ‖g‖, ‖g‖²
        xdiff = xnorm = None
        if _sw2cd:                                                                          # 2c/2d: eigenspace-restricted squared-projection Rayleigh
            pairs = _eds_pairs(lambda v: hvp_S(model, th_c, X, rr_c.reshape(N, outD), v), swmrk)  # (σ_i,u_i) of M_r=Σ_k r_kQ_k
            Ap = sum(lam * (float(u @ g) ** 2) for (lam, u) in pairs if lam > 0) / g2       # A₊ = Σ_{σ_i>0} σ_i·(u_iᵀg)² / ‖g‖²  (≥0)
            An = sum(-lam * (float(u @ g) ** 2) for (lam, u) in pairs if lam < 0) / g2      # A₋ = Σ_{σ_i<0} |σ_i|·(u_iᵀg)² / ‖g‖²  (≥0; squared ⇒ gauge-invariant)
            xden = Ap + An; xdiff = Ap - An; xnorm = (xdiff / xden) if xden > 1e-30 else 0.0  # 2c x = A₊−A₋; 2d x = (A₊−A₋)/(A₊+A₋)
        x2e = None
        if _sw2e:                                                                           # 2e: FULL-operator quadratic form gᵀM_r g (one HVP)
            Mr_g = hvp_S(model, th_c, X, rr_c.reshape(N, outD), g)                          # M_r·g = (Σ_k r_kQ_k) g
            _x2e = float(g @ Mr_g); x2e = _x2e if _x2e == _x2e else None                    # x = gᵀ M_r g = (Jᵀr)ᵀ M_r (Jᵀr)
        th_g = th_c.clone(); ok = True                                                      # y: short-horizon Δ‖J‖_F over l GD steps (config lr) — SHARED by 2c & 2e
        for _ in range(_ljl):
            _og = model.forward(th_g, X)
            if not (float(loss.value(_og, Y, N)) < 1e12):
                ok = False; break                                                          # diverged within l steps ⇒ no y
            th_g = th_g - _lgd * grad_loss(model, loss, th_g, X, Y)[0]
        dj = djn = None
        if ok:
            Jcl, _ = jac_cols(model, th_g, X); jl = float(Jcl[:M].norm())                  # ‖J_l‖_F
            if jl == jl:
                dj = jl - j0; djn = (jl - j0) / (j0 + _EPS2D)
        yield {"type": "sweeppt2c", "i": si, "std": float(std_c),
               "xdiff": (xdiff if (xdiff is not None and xdiff == xdiff) else None),
               "xnorm": (xnorm if (xnorm is not None and xnorm == xnorm) else None),
               "x2e": x2e,
               "dj": (dj if (dj is not None and dj == dj) else None),
               "djn": (djn if (djn is not None and djn == djn) else None),
               "l": _ljl, "swmrk": swmrk, "n2c": npairs}

    for pi in range(npairs):
        lr = math.exp(rng.uniform(math.log(lrmin), math.log(lrmax)))
        std = math.exp(rng.uniform(math.log(stdmin), math.log(stdmax)))
        th = model.init_theta(seed0 + 1, std)
        _sharp0 = _sharp_top(lambda v: hvp_L(model, loss, th, X, Y, v))   # λmax(∇²L) at init — EoS guard for prediction-2/2b (plots 3-4)
        lr_eff = lr; eos_drop = False; rescued = False
        if _sharp0 is not None and _sharp0 > 0.0 and _sharp0 >= 2.0 / lr:   # starts at/above EoS (2/lr ≤ sharpness₀) ⇒ rescue by lowering lr below EoS, else drop from plots 3-4
            _lr_resc = 0.9 * (2.0 / _sharp0)                               # 2/lr' = sharpness₀/0.9 > sharpness₀ ⇒ starts below EoS
            if _lr_resc >= lrmin:
                lr_eff = _lr_resc; rescued = True                          # RESCUE: lower lr (affects the shared trajectory → all 4 row-1 plots)
            else:
                eos_drop = True                                            # cannot rescue within the sweep's lr range ⇒ exclude from plots 3-4
        ps5x = None; ps5t = None                                          # magnified-PS: Σ u₁ᵀ(QJr)v₁ frozen at the PS-onset step (None ⇒ pair never enters PS below EoS)
        diffs = []; losses = []; jnorms = []; th_at_T = None; diverged = False; align_acc = 0.0; align_cnt = 0
        for t in range(steps):
            out = model.forward(th, X)
            lv = float(loss.value(out, Y, N))
            if not (lv < 1e12):                            # NaN or blow-up ⇒ stop this pair, keep the finite history
                diverged = True
                break
            Jc, _ = jac_cols(model, th, X); Jm = Jc[:M]
            jnorms.append(float(Jm.norm()))                # ‖J‖_F per step → start-of-training trend (prediction-2b)
            cS = loss.resid_cotangent(out.reshape(N, outD), Y, N)
            rr = (-N * cS).reshape(-1)[:M]
            if t < swsumn:                                 # prediction-2/2b tick colour: Σ_{k≤3}|⟨r̂, v_k⟩| (top-3 NTK eigvecs), averaged over the window
                try:
                    _Wv, _Vv = torch.linalg.eigh(Jm @ Jm.t())
                    _rn = rr / max(float(rr.norm()), 1e-30)
                    align_acc += float(sum(abs(float(_rn @ _Vv[:, -1 - _k])) for _k in range(min(3, M)))); align_cnt += 1
                except Exception:
                    pass
            if opt == "gd":
                gL = grad_loss(model, loss, th, X, Y)[0]   # GD: exact ∇L path (byte-identical to the original)
                jrd = float((Jm.t() @ (Jm @ gL)).norm())
                jdr = float(hvp_S(model, th, X, rr.reshape(N, outD), gL).norm())
                dth_sw = -lr_eff * gL                       # EoS-guard: trajectory uses the (possibly rescued) lr_eff
            else:
                g, scale = precond_flow(model, opt, Jm, rr, N, lr_eff)   # preconditioned actual trajectory (direction, step)
                jrd = float((Jm.t() @ (Jm @ g)).norm())
                jdr = float(hvp_S(model, th, X, rr.reshape(N, outD), g).norm())
                dth_sw = scale * g
            diffs.append(jrd - jdr); losses.append(lv)
            if (ps5x is None) and (jrd > jdr) and (_sharp0 is not None) and (_sharp0 < 2.0 / lr_eff):   # magnified-PS: freeze at the FIRST PS-onset step (t=0 if starts in PS, else the neg→pos jrd−jdr crossing — matching Pred-3) AND below EoS
                try:
                    _Kw, _Ku = torch.linalg.eigh(Jm @ Jm.t())               # NTK = J₀J₀ᵀ (M×M)
                    _u1 = _Ku[:, -1]                                        # top NTK eigenvector
                    _v1 = Jm.t() @ _u1; _v1 = _v1 / max(float(_v1.norm()), 1e-30)   # top right-singular vector of J₀
                    _c1 = _u1.reshape(N, outD); _rc = rr.clone(); _acc = 0.0
                    for _tt in range(swps5n):
                        _Jr = Jm.t() @ _rc                                  # J₀ᵀ r_t
                        _acc += float(_v1 @ hvp_S(model, th, X, _c1, _Jr))   # u₁ᵀ(QJr_t)v₁ = v₁ᵀ Σ_k(u₁)_k Q_k (J₀ᵀr_t); server hvpS(th,X,_Jr,_c1) → eos hvp_S(...,_c1,_Jr)
                        _rc = _rc - (lr_eff / N) * (Jm @ _Jr)               # r_{t+1}=r_t−(η/N)J₀(J₀ᵀr_t) (frozen-J NTK residual dynamics)
                    ps5x = _acc; ps5t = t                                    # record the PS-onset step
                except Exception:
                    ps5x = None
            if t == Tstart:
                th_at_T = th.detach().clone()
            th = th + dth_sw                                # actual trajectory follows the chosen optimizer
        tAct, censA = first_signchange(diffs, steps)
        if th_at_T is not None:                            # predict only if θ(Tstart) reached before divergence
            tPredRel, censP = pred_signchange(model, loss, th_at_T, X, Y, N, outD, p, float(lr_eff), K, steps - Tstart, cubic=True, opt=opt)
            tPred = tPredRel + Tstart
            tPredQRel, censPQ = pred_signchange(model, loss, th_at_T, X, Y, N, outD, p, float(lr_eff), K, steps - Tstart, cubic=False, opt=opt)   # prediction-1 plot 1: quadratic (T=0, Q fixed)
            tPredQ = tPredQRel + Tstart
        else:
            tPred, censP = steps, True; tPredQ, censPQ = steps, True
        alignNTK = float(align_acc / align_cnt) if align_cnt else 0.0   # residual↔top-3-NTK-eigvec alignment (LEFT-half tick colour for prediction-2/2b)
        curv = max_abs_curvature(losses)                   # peak signed d²loss/dt² over the finite prefix (legacy)
        sumD = float(sum(diffs[:swsumn]))                  # prediction-2: Σ d = Σ(jrd−jdr) over the first swsumn iterations (from 0)
        sumCurv = float(sum(curvature_at(losses, tt) for tt in range(min(swsumn, len(losses)))))   # prediction-2: Σ d²loss/dt² over the first swsumn iterations (from 0)
        sumD2 = float(sum(diffs[:swsumn2]))                # prediction-2b: Σ d over the first swsumn2 iterations (own window)
        nj2 = min(swsumn2, len(jnorms) - 1)                # prediction-2b: Σ ∇‖J‖ over the first swsumn2 iters = ‖J‖_{swsumn2} − ‖J‖_0 (telescoping)
        sumJgrad = float(sum(jnorms[i + 1] - jnorms[i] for i in range(nj2))) if nj2 >= 1 else 0.0
        jfro = float(sum(jnorms[:swsumn]) / max(1, min(swsumn, len(jnorms))))   # windowed-mean ‖J‖_F over the first swsumn steps — RIGHT-half tick colour for prediction-2/2b
        yield {"type": "sweeppt", "i": pi, "lr": float(lr_eff), "std": float(std),
               "lrOrig": float(lr), "sharp0": (float(_sharp0) if (_sharp0 is not None and _sharp0 == _sharp0) else None),   # EoS guard: λmax(∇²L) at init + the pair's original swept lr (before any rescue)
               "eosDrop": bool(eos_drop), "rescued": bool(rescued),   # eosDrop ⇒ started at EoS & un-rescuable ⇒ exclude from plots 3-4; rescued ⇒ lr was lowered to start below EoS
               "tAct": tAct, "censA": censA or diverged, "tPred": tPred, "censP": censP,
               "tPredQ": tPredQ, "censPQ": censPQ,                                   # prediction-1 plot 1: quadratic (T=0, Q fixed) predicted t*
               "alignNTK": (alignNTK if alignNTK == alignNTK else 0.0),             # Σ|⟨r̂,v_k⟩| top-3 NTK — prediction-2/2b LEFT-half tick colour
               "jfro": (jfro if jfro == jfro else 0.0),                             # windowed-mean ‖J‖_F — prediction-2/2b RIGHT-half tick colour
               "sumD": (sumD if sumD == sumD else 0.0), "sumCurv": (sumCurv if sumCurv == sumCurv else 0.0),
               "sumD2": (sumD2 if sumD2 == sumD2 else 0.0), "sumJgrad": (sumJgrad if sumJgrad == sumJgrad else 0.0),
               "ps5x": (ps5x if (ps5x is not None and ps5x == ps5x) else None), "ps5t": ps5t, "swps5n": swps5n,   # magnified-PS: Σ u₁ᵀ(QJr)v₁ (x) frozen at PS-onset step ps5t; None ⇒ pair never enters PS below EoS. y = sumJgrad
               "curv": (curv if curv == curv else 0.0), "swsumn": swsumn, "swsumn2": swsumn2, "diverged": diverged}


# ═══════════════════════════════════════════════════════════════════════ per-step forecast state machines
class Pred3Tracker:
    """Prediction-3 (server s36): after (jrd−jdr)=‖J·ṙ‖−‖J̇·r‖ first changes sign at t*, freeze J*=J(t*) and
    compare the ACTUAL residual with three theories propagated from r(t*): (1) a frozen-J linear NTK theory
    r_{k+1}=(I−scale·J*·P·J*ᵀ)r_k, (2) a 2nd-order theory that adds −½·scale²·(gᵀQ*g), and (3) an EVOLVING-J
    step-function theory — J is held fixed for p3k iters while r decays, then the window-J JUMPS to a shadow
    that advanced every step via the TRUE QJr recurrence. All theory steps follow the run's `optimizer` via
    models.precond_flow (gd ⇒ plain (Jᵀr, η/N) GD). With p3rep>0 every theory RE-anchors from the live run
    every p3rep iters. Also drives the MULTI-FREEZE panel (J,Q frozen at t*−10 / t*+10 / t*+20, back-filled).

    MIRRORS server.py's g_pred3 + g_pred3m per-step blocks (server.py:3766-3868) EXACTLY. Two records per
    step: `step()` returns g_pred3 (None until Jc/rr are available), `step_multi()` returns g_pred3m.

    The tracker owns the sign-change state (t*, frozen J*/θ*, the propagated theories), so t* found in `step`
    is visible to `step_multi` in the same tick — mirroring the server, where both blocks share p3_tstar.
    """

    def __init__(self, model, loss, N, outD, p, lr, ee, opt="gd", p3rep=100, p3k=10):
        self.model = model; self.loss = loss
        self.N = N; self.outD = outD; self.p = p; self.M = N * outD
        self.lr = lr; self.ee = max(1, ee)
        self.opt = opt; self.p3rep = int(p3rep); self.p3k = max(1, int(p3k))
        # single-freeze state (server: p3_*): sign tracker, t*, frozen J*/θ*, 1st/2nd-order theory residuals
        self.p3_prev_sign = None; self.p3_tstar = None
        self.p3_Jstar = None; self.p3_rtheory = None
        self.p3_thstar = None; self.p3_r2 = None
        # 3rd (evolving-J step-function) theory state + repeat-anchor time
        self.p3_J3 = None; self.p3_J3adv = None; self.p3_r3 = None; self.p3_the3 = None; self.p3_j3sc = 0
        self.p3_anchor_t = None
        # multi-freeze state (server: p3m)
        self.p3m = None
        # cached per-tick Jc/rr slices (set in step(), reused by step_multi())
        self._Jm3 = None; self._rm3 = None; self._th = None

    def _adv3m(self, Js, ths, r1, r2, r3, J3, J3adv, sc3, X):
        """One optimizer step of the multi-freeze theories: 1st & 2nd (frozen J*=Js, Q frozen at ths=θ*) + 3rd
        (evolving-J STEP function; shadow J3adv advances every step via QJr, window-J3 commits every p3k iters).
        MIRRORS server _adv3m (server.py:3822-3839)."""
        N = self.N; lr = self.lr; M = self.M; p3k = self.p3k
        g1m, s1m = precond_flow(self.model, self.opt, Js, r1, N, lr)       # 1st: ṙ=−scale·J*·(P·J*ᵀr); gd ⇒ (J*ᵀr, η/N)
        r1n = r1 - s1m * (Js @ g1m)                                        # r=(I−scale·J*·P·J*ᵀ)r
        g2, s2m = precond_flow(self.model, self.opt, Js, r2, N, lr)        # 2nd order (J*,Q* frozen at t*): g2=P·(J*ᵀr₂)
        quad2 = 0.5 * (s2m ** 2) * (jac_hvp(self.model, ths, X, g2)[:M] * g2).sum(dim=1)   # ½·scale²·(g2ᵀQ*_k g2)_k
        r2n = r2 - s2m * (Js @ g2) - quad2    # 2nd: curvature enters with a MINUS (Δr=−Δf)
        r3n = r3; J3adv_n = J3adv                                          # 3rd (evolving-J STEP function)
        if torch.isfinite(J3).all():
            if J3adv_n is None:
                J3adv_n = J3.detach().clone()
            gshm, sshm = precond_flow(self.model, self.opt, J3adv_n, r3, N, lr)   # shadow J̇ = scale·Q(ths)·(P·J3advᵀr3)
            J3adv_n = J3adv_n + sshm * jac_hvp(self.model, ths, X, gshm)[:M]       # QJr EVERY step (Q frozen at ths), using r_t
            g3m, s3m = precond_flow(self.model, self.opt, J3, r3, N, lr)   # g = P·(J3ᵀ r3) with the CONSTANT window-J3
            r3n = r3 - s3m * (J3 @ g3m)                                    # r3 decays via the CONSTANT window-J3
            sc3 += 1
            if sc3 % p3k == 0:
                J3 = J3adv_n.detach().clone(); J3adv_n = J3.detach().clone()   # COMMIT window-J3 to the p3k-step shadow, restart shadow
        return r1n, r2n, r3n, J3, J3adv_n, sc3

    def step(self, t, th, X, Y, Jc, rr):
        """g_pred3 record for this tick. Returns None when Jc/rr are unavailable (multi-sample gate off)."""
        model = self.model; loss = self.loss
        N = self.N; M = self.M; lr = self.lr; ee = self.ee; p3k = self.p3k
        if Jc is None or rr is None:
            self._Jm3 = self._rm3 = self._th = None
            return None
        Jm3 = Jc[:M]; rm3 = rr[:M]
        self._Jm3, self._rm3, self._th = Jm3, rm3, th          # share with step_multi
        gL3 = grad_loss(model, loss, th, X, Y)[0]
        jrd3 = float((Jm3.t() @ (Jm3 @ gL3)).norm())            # ‖J·ṙ‖ = ‖JᵀJ∇L‖
        jdr3 = float(hvp_S(model, th, X, rm3.reshape(N, self.outD), gL3).norm())   # ‖J̇·r‖ = ‖M_r∇L‖
        diff3 = jrd3 - jdr3
        sgn3 = (1 if diff3 > 0 else (-1 if diff3 < 0 else 0))
        g_pred3 = {"jrd": jrd3, "jdr": jdr3, "diff": diff3}
        if self.p3_tstar is None and self.p3_prev_sign == -1 and sgn3 == 1:   # anchor SPECIFICALLY on the NEG→POS crossing of diff3=‖J·ṙ‖−‖J̇·r‖ (‖J̇·r‖−‖J·ṙ‖ becoming negative) — mirrors server.run_stream
            self.p3_tstar = t
            self.p3_Jstar = Jm3.detach().clone(); self.p3_rtheory = rm3.detach().clone()   # freeze J*, r_theory = r(t*)
            self.p3_thstar = th.detach().clone(); self.p3_r2 = rm3.detach().clone()         # freeze θ* (for Q*), r₂ = r(t*)
            self.p3_J3 = Jm3.detach().clone(); self.p3_J3adv = None; self.p3_r3 = rm3.detach().clone()
            self.p3_the3 = th.detach().clone(); self.p3_j3sc = 0                             # 3rd theory: seed J3=J(t*), r3=r(t*), θ̂3=θ(t*)
            self.p3_anchor_t = t
        elif self.p3_tstar is not None and self.p3rep > 0 and self.p3_anchor_t is not None and t >= self.p3_anchor_t + self.p3rep:
            self.p3_Jstar = Jm3.detach().clone(); self.p3_rtheory = rm3.detach().clone()     # ★ repeat_iter: RE-anchor ALL theories from the LIVE run every p3rep iters
            self.p3_thstar = th.detach().clone(); self.p3_r2 = rm3.detach().clone()
            self.p3_J3 = Jm3.detach().clone(); self.p3_J3adv = None; self.p3_r3 = rm3.detach().clone()
            self.p3_the3 = th.detach().clone(); self.p3_j3sc = 0
            self.p3_anchor_t = t
        if sgn3 != 0:
            self.p3_prev_sign = sgn3
        if self.p3_tstar is not None:
            na3 = float(rm3.norm()); nt3 = float(self.p3_rtheory.norm()); nt3b = float(self.p3_r2.norm()); nt3c = float(self.p3_r3.norm())
            cos3 = (float(rm3 @ self.p3_rtheory) / (na3 * nt3)) if (na3 > 1e-30 and nt3 > 1e-30) else None
            cos3b = (float(rm3 @ self.p3_r2) / (na3 * nt3b)) if (na3 > 1e-30 and nt3b > 1e-30) else None
            cos3c = (float(rm3 @ self.p3_r3) / (na3 * nt3c)) if (na3 > 1e-30 and nt3c > 1e-30) else None   # evolving-J theory direction cosine
            g_pred3.update({"tstar": self.p3_tstar, "rAct": na3, "rThy": nt3, "cos": cos3,
                            "rThy2": nt3b, "cos2": cos3b, "rThy3": nt3c, "cos3": cos3c})
            for _ in range(max(1, ee)):   # advance the 1st + 2nd + 3rd theories ONE GD step per actual step
                g1p, sc1p = precond_flow(model, self.opt, self.p3_Jstar, self.p3_rtheory, N, lr)   # 1st order
                self.p3_rtheory = self.p3_rtheory - sc1p * (self.p3_Jstar @ g1p)
                g2, sc2 = precond_flow(model, self.opt, self.p3_Jstar, self.p3_r2, N, lr)          # 2nd order (J*,Q* frozen at t*)
                quad2 = 0.5 * (sc2 ** 2) * (jac_hvp(model, self.p3_thstar, X, g2)[:M] * g2).sum(dim=1)
                self.p3_r2 = self.p3_r2 - sc2 * (self.p3_Jstar @ g2) - quad2                        # curvature enters with a MINUS (Δr=−Δf)
                if torch.isfinite(self.p3_J3).all() and torch.isfinite(self.p3_the3).all():         # 3rd theory (J EVOLVING as a STEP function)
                    if self.p3_J3adv is None:
                        self.p3_J3adv = self.p3_J3.detach().clone()
                    gsh, scsh = precond_flow(model, self.opt, self.p3_J3adv, self.p3_r3, N, lr)      # shadow J̇ = scale·Q(θ̂3)·(P·J3advᵀr3)
                    self.p3_J3adv = self.p3_J3adv + scsh * jac_hvp(model, self.p3_the3, X, gsh)[:M]  # QJr EVERY step (Q at θ̂3), using r_t
                    g3e, sc3e = precond_flow(model, self.opt, self.p3_J3, self.p3_r3, N, lr)         # g = P·(J3ᵀ r3) with the CONSTANT window-J3
                    self.p3_r3 = self.p3_r3 - sc3e * (self.p3_J3 @ g3e)                              # r3 decays via the CONSTANT window-J3
                    self.p3_the3 = self.p3_the3 + sc3e * g3e                                         # θ̂3 self-trajectory (where Q is re-evaluated)
                    self.p3_j3sc += 1
                    if self.p3_j3sc % p3k == 0:                                                      # COMMIT: window-J3 jumps to the p3k-step shadow
                        self.p3_J3 = self.p3_J3adv.detach().clone()
                        self.p3_J3adv = self.p3_J3.detach().clone()
        return g_pred3

    def step_multi(self, t, th, X, Y, Jc, rr):
        """g_pred3m record (multi-freeze). Call AFTER step() (shares the frozen-J* state + cached Jm3/rm3).
        Returns None until t* has occurred. MIRRORS server.py's g_pred3m block (server.py:3813-3868)."""
        N = self.N; M = self.M; lr = self.lr; ee = self.ee
        if Jc is None or rr is None:
            return None
        Jm3 = self._Jm3 if self._Jm3 is not None else Jc[:M]
        rm3 = self._rm3 if self._rm3 is not None else rr[:M]
        if self.p3m is None:
            self.p3m = {"buf": [], "th": [None, None, None]}       # rolling (t,J,r,‖r‖,θ) buffer + 3 frozen theories
        p3m = self.p3m
        p3m["buf"].append((t, Jm3.detach().clone(), rm3.detach().clone(), float(rm3.norm()), th.detach().clone()))
        if len(p3m["buf"]) > 30:
            p3m["buf"] = p3m["buf"][-30:]
        offs = [-10, 10, 20]; backfill = None
        if self.p3_tstar is not None and p3m["th"][0] is None:      # theory-0 (t*−10): freeze from the buffer, BACK-FILL t*−10 … t
            tgt = self.p3_tstar - 10
            bi = min(range(len(p3m["buf"])), key=lambda i: abs(p3m["buf"][i][0] - tgt))
            J0s = p3m["buf"][bi][1]; th0s = p3m["buf"][bi][4]
            rth = p3m["buf"][bi][2].clone(); r2th = p3m["buf"][bi][2].clone()
            r3th = p3m["buf"][bi][2].clone(); J3th = J0s.detach().clone(); J3adv_th = None; sc3 = 0; bf = []
            for j in range(bi, len(p3m["buf"])):
                bf.append([p3m["buf"][j][0], p3m["buf"][j][3], float(rth.norm()), float(r2th.norm()), float(r3th.norm())])   # (step, ‖r_act‖, 1st, 2nd, 3rd)
                nadv = (p3m["buf"][j + 1][0] - p3m["buf"][j][0]) if j + 1 < len(p3m["buf"]) else max(1, ee)   # buffer points are eigevery GD-steps apart → advance that many; last → ee to hand off
                for _ in range(max(1, nadv)):
                    rth, r2th, r3th, J3th, J3adv_th, sc3 = self._adv3m(J0s, th0s, rth, r2th, r3th, J3th, J3adv_th, sc3, X)
            p3m["th"][0] = {"J": J0s, "th": th0s, "r": rth, "r2": r2th, "r3": r3th, "J3": J3th, "J3adv": J3adv_th, "sc3": sc3, "anchor_t": t}; backfill = bf
        for k in (1, 2):                                            # theory-1/2 (t*+10, t*+20): freeze when the run reaches them
            if self.p3_tstar is not None and p3m["th"][k] is None and t >= self.p3_tstar + offs[k]:   # >= (not ==): eigevery>1 skips the exact offset tick
                p3m["th"][k] = {"J": Jm3.detach().clone(), "th": th.detach().clone(), "r": rm3.detach().clone(),
                                "r2": rm3.detach().clone(), "r3": rm3.detach().clone(), "J3": Jm3.detach().clone(),
                                "J3adv": None, "sc3": 0, "anchor_t": t}
        thy = [None, None, None]; thy2 = [None, None, None]; thy3 = [None, None, None]
        for k in range(3):
            if p3m["th"][k] is None or (k == 0 and backfill is not None):
                continue                                           # skip theory-0's single emit on the back-fill tick
            d = p3m["th"][k]
            if self.p3rep > 0 and d.get("anchor_t") is not None and t >= d["anchor_t"] + self.p3rep:   # ★ repeat_iter: RE-anchor this frozen theory from the LIVE run
                d["r"] = rm3.detach().clone(); d["r2"] = rm3.detach().clone(); d["r3"] = rm3.detach().clone()
                d["J"] = Jm3.detach().clone(); d["J3"] = Jm3.detach().clone(); d["J3adv"] = None
                d["th"] = th.detach().clone(); d["sc3"] = 0; d["anchor_t"] = t
            thy[k] = float(d["r"].norm()); thy2[k] = float(d["r2"].norm()); thy3[k] = float(d["r3"].norm())
            for _ in range(max(1, ee)):
                d["r"], d["r2"], d["r3"], d["J3"], d["J3adv"], d["sc3"] = self._adv3m(
                    d["J"], d["th"], d["r"], d["r2"], d["r3"], d["J3"], d.get("J3adv"), d["sc3"], X)
        if self.p3_tstar is not None:
            return {"rAct": float(rm3.norm()), "thy": thy, "thy2": thy2, "thy3": thy3,
                    "off": offs, "tstar": self.p3_tstar, "backfill": backfill}
        return None


class Pred4Tracker:
    """Prediction-4 (server s37): frozen-M_r Jacobian propagation. At iteration t0 freeze θ0, J0=J(t0) and the
    residual r0; then propagate J_{s+1}=(I+scale·M_r)J_s with M_r=Σ_k r0_k Q_k(θ0) frozen (matrix-free via
    hvp_S), and compare the top-3 NTK eigenvalues λ(JJᵀ) of predicted vs actual J. A second EVOLVING-Qr variant
    propagates Ĵ_{t+1}=Ĵ_t+scale·M_r(r̂Q,θ̂Q)·Ĵ where rQ is FIXED for p4s-step windows (a true (I+scale·M_r)^s
    power iteration), θ̂ is a GD/optimizer self-trajectory (via precond_flow) and r̂ is a BOUNDED quadratic
    self-residual Y−(f0+J0Δθ̂+½Δθ̂ᵀQ0Δθ̂); a trust-region clamp (‖g‖≤1e4) + norm cap (‖Ĵ‖<1e8) keep GN's big
    steps from overflowing the eigh. `scale = lr/N` (gd) or `lr` (preconditioned). With p4rep>0 the whole
    propagation RE-anchors from the live run every p4rep iters. Also drives the MULTI-FREEZE panel (t0−10/+10/+20).

    MIRRORS server.py's g_pred4 + g_pred4m per-step blocks (server.py:3870-3975) EXACTLY. `step()` returns
    g_pred4 (with kAct/kPred/kPredE + cos5/cos5E/cosGN5/cosGN5E), `step_multi()` returns g_pred4m.
    """

    def __init__(self, model, loss, N, outD, p, lr, ee, p4t0=10, p4s=10, p4rep=100, opt="gd"):
        self.model = model; self.loss = loss
        self.N = N; self.outD = outD; self.p = p; self.M = N * outD
        self.lr = lr; self.ee = max(1, ee)
        self.p4t0 = max(0, int(p4t0)); self.p4s = max(1, int(p4s)); self.p4rep = int(p4rep); self.opt = opt
        # single-freeze state (server: p4_*): frozen-M_r J + the EVOLVING-Qr self-model (Ĵ, r̂, θ̂, fixed-rQ window)
        self.p4_J = None; self.p4_th0 = None; self.p4_r0 = None
        self.p4_Je = None; self.p4_re = None; self.p4_J0e = None
        self.p4_the = None; self.p4_thQ = None; self.p4_reQ = None; self.p4_qsc = 0
        self.p4_f0 = None; self.p4_anchor_t = None
        # multi-freeze state (server: p4m)
        self.p4m = None

    def step(self, t, th, X, Y, Jc, rr):
        model = self.model
        N = self.N; M = self.M; lr = self.lr; ee = self.ee; outD = self.outD
        p4t0 = self.p4t0; p4s = self.p4s; p4rep = self.p4rep
        if Jc is None or rr is None:
            return None
        Jm4 = Jc[:M]
        if (self.p4_J is None and t >= p4t0) or (self.p4_J is not None and p4rep > 0 and self.p4_anchor_t is not None and t >= self.p4_anchor_t + p4rep):   # freeze Q & residual at t0, then ★ RE-anchor from the LIVE run every p4rep iters
            self.p4_J = Jm4.detach().clone(); self.p4_th0 = th.detach().clone(); self.p4_r0 = rr[:M].reshape(N, outD).detach().clone()
            self.p4_Je = Jm4.detach().clone(); self.p4_re = rr[:M].detach().clone()   # EVOLVING-Qr: seed Ĵ, r̂
            self.p4_J0e = Jm4.detach().clone()                                        # frozen J(θ0) for the bounded quadratic self-residual
            self.p4_the = th.detach().clone(); self.p4_thQ = th.detach().clone(); self.p4_reQ = None; self.p4_qsc = 0   # θ̂ (GD-propagated); θ̂Q/r̂Q FIXED for the s-step rQ window
            self.p4_f0 = (Y.reshape(-1)[:M] - rr[:M]).detach().clone()                # f(θ0) anchor for the bounded quadratic self-residual
            self.p4_anchor_t = t
        if self.p4_J is None:
            return None
        K4 = min(3, M); K5 = min(3, M)                             # top-3 everywhere
        try:                                                       # eigh → eigenvalues AND eigenvectors (top-3 cosines)
            Wa, Va = torch.linalg.eigh(Jm4 @ Jm4.t())              # actual NTK: eigvals asc, eigvecs = columns
            Wp, Vp = torch.linalg.eigh(self.p4_J @ self.p4_J.t())  # predicted NTK (frozen residual)
            We, Ve = torch.linalg.eigh(self.p4_Je @ self.p4_Je.t())  # predicted NTK (evolving residual)
            kAct4 = [max(0.0, float(Wa[-1 - i])) for i in range(K4)]
            kPred4 = [max(0.0, float(Wp[-1 - i])) for i in range(K4)]
            kPredE = [max(0.0, float(We[-1 - i])) for i in range(K4)]
            cos5 = [min(1.0, abs(float(Va[:, -1 - i] @ Vp[:, -1 - i]))) for i in range(K5)]    # |cos| actual vs frozen-residual NTK eigvec (clamp: fp32 can give 1+2⁻²¹)
            cos5E = [min(1.0, abs(float(Va[:, -1 - i] @ Ve[:, -1 - i]))) for i in range(K5)]   # |cos| actual vs evolving-residual NTK eigvec
            cosGN5 = []; cosGN5E = []                              # |cos| of the GN (JᵀJ) eigvecs = right-singular vecs z=Jᵀu/‖·‖
            for i in range(K5):
                za = Jm4.t() @ Va[:, -1 - i]; za = za / max(float(za.norm()), 1e-30)
                zp = self.p4_J.t() @ Vp[:, -1 - i]; zp = zp / max(float(zp.norm()), 1e-30)
                ze = self.p4_Je.t() @ Ve[:, -1 - i]; ze = ze / max(float(ze.norm()), 1e-30)
                cosGN5.append(min(1.0, abs(float(za @ zp)))); cosGN5E.append(min(1.0, abs(float(za @ ze))))
        except Exception:                                          # fp32-GPU eigh hiccup ⇒ eigenvalues only
            ka = safe_eigvalsh(Jm4 @ Jm4.t()); kp = safe_eigvalsh(self.p4_J @ self.p4_J.t())
            kpe = safe_eigvalsh(self.p4_Je @ self.p4_Je.t())
            kAct4 = [max(0.0, float(ka[-1 - i])) for i in range(K4)]
            kPred4 = [max(0.0, float(kp[-1 - i])) for i in range(K4)]
            kPredE = [max(0.0, float(kpe[-1 - i])) for i in range(K4)]
            cos5 = []; cos5E = []; cosGN5 = []; cosGN5E = []
        g_pred4 = {"t0": p4t0, "s": p4s, "kAct": kAct4, "kPred": kPred4, "kPredE": kPredE,
                   "cos5": cos5, "cos5E": cos5E, "cosGN5": cosGN5, "cosGN5E": cosGN5E}
        sc4 = lr / max(N, 1) if self.opt == "gd" else lr          # optimizer step size (gd ⇒ η/N)
        for _ in range(max(1, ee)):                               # advance ONE GD step per actual step
            if torch.isfinite(self.p4_J).all() and float(self.p4_J.norm()) < 1e8:   # norm cap ⇒ GN divergence stays finite (no eigh stall)
                MrJ = torch.stack([hvp_S(model, self.p4_th0, X, self.p4_r0, self.p4_J[k]) for k in range(M)])   # frozen M_r·J
                self.p4_J = self.p4_J + sc4 * MrJ
            if torch.isfinite(self.p4_Je).all() and torch.isfinite(self.p4_re).all() and torch.isfinite(self.p4_the).all() and float(self.p4_Je.norm()) < 1e8:   # EVOLVING-Qr self-model
                Yf = Y.reshape(-1)[:M]
                if self.p4_reQ is None or self.p4_qsc % p4s == 0:                    # refresh the FIXED rQ=M_r(r̂Q,θ̂Q) at each s-step window boundary
                    self.p4_thQ = self.p4_the.detach().clone(); self.p4_reQ = self.p4_re.detach().clone()
                self.p4_Je = self.p4_Je + sc4 * torch.stack([hvp_S(model, self.p4_thQ, X, self.p4_reQ.reshape(N, outD), self.p4_Je[k]) for k in range(M)])   # Ĵ_{t+1}=Ĵ+scale·M_r(r̂Q,θ̂Q)·Ĵ (true (I+scale·M_r)^s)
                g_e = precond_flow(self.model, self.opt, self.p4_Je, self.p4_re, N, lr)[0]   # preconditioned self-model gradient P·(Ĵᵀr̂); gd ⇒ Ĵᵀr̂
                _neg = float(g_e.norm())
                if _neg > 1e4: g_e = g_e * (1e4 / _neg)                              # trust-region clamp (GN's Newton step can diverge the frozen-quad extrapolation)
                self.p4_the = self.p4_the + sc4 * g_e                                # θ̂ self-trajectory (follows the optimizer)
                dth_e = self.p4_the - self.p4_th0
                HzD_e = jac_hvp(model, self.p4_th0, X, dth_e)[:M]                     # Q0·Δθ̂
                self.p4_re = Yf - (self.p4_f0 + self.p4_J0e @ dth_e + 0.5 * (HzD_e @ dth_e))   # r̂ = Y − (f0 + J0Δθ̂ + ½Δθ̂ᵀQ0Δθ̂)
                self.p4_qsc += 1
        return g_pred4

    def step_multi(self, t, th, X, Y, Jc, rr):
        model = self.model
        N = self.N; M = self.M; lr = self.lr; ee = self.ee; outD = self.outD
        p4t0 = self.p4t0; p4s = self.p4s; p4rep = self.p4rep
        if Jc is None or rr is None:
            return None
        Jm4m = Jc[:M]
        if self.p4m is None:
            self.p4m = [{"tf": p4t0 - 10, "J": None, "th0": None, "r0": None, "Je": None, "re": None, "the": None, "thQ": None, "reQ": None, "J0e": None, "f0": None, "qsc": 0, "anchor_t": None},
                        {"tf": p4t0 + 10, "J": None, "th0": None, "r0": None, "Je": None, "re": None, "the": None, "thQ": None, "reQ": None, "J0e": None, "f0": None, "qsc": 0, "anchor_t": None},
                        {"tf": p4t0 + 20, "J": None, "th0": None, "r0": None, "Je": None, "re": None, "the": None, "thQ": None, "reQ": None, "J0e": None, "f0": None, "qsc": 0, "anchor_t": None}]
        K5m = min(3, M)                                            # top-3
        try:
            Wam = safe_eigvalsh(Jm4m @ Jm4m.t()); ka4m = [max(0.0, float(Wam[-1 - i])) for i in range(K5m)]
        except Exception:
            ka4m = None
        kpm = [None, None, None]; kpmE = [None, None, None]
        for fi4, fz in enumerate(self.p4m):
            if (fz["tf"] >= 0 and t >= fz["tf"] and fz["J"] is None) or (fz["J"] is not None and p4rep > 0 and fz["anchor_t"] is not None and t >= fz["anchor_t"] + p4rep):   # freeze at this offset (>= so eigevery>1 can't skip the exact tick), then ★ RE-anchor from the LIVE run every p4rep iters
                fz["J"] = Jm4m.detach().clone(); fz["th0"] = th.detach().clone(); fz["r0"] = rr[:M].reshape(N, outD).detach().clone()
                fz["Je"] = Jm4m.detach().clone(); fz["re"] = rr[:M].detach().clone()   # evolving-Qr seed
                fz["J0e"] = Jm4m.detach().clone()                          # frozen J(θ_freeze) for the bounded quadratic self-residual
                fz["the"] = th.detach().clone(); fz["thQ"] = th.detach().clone(); fz["reQ"] = None; fz["qsc"] = 0
                fz["f0"] = (Y.reshape(-1)[:M] - rr[:M]).detach().clone()   # f(θ_freeze) for the bounded quadratic self-residual
                fz["anchor_t"] = t
            if fz["J"] is not None:
                try:
                    Wpm = safe_eigvalsh(fz["J"] @ fz["J"].t()); kpm[fi4] = [max(0.0, float(Wpm[-1 - i])) for i in range(K5m)]
                    Wpe = safe_eigvalsh(fz["Je"] @ fz["Je"].t()); kpmE[fi4] = [max(0.0, float(Wpe[-1 - i])) for i in range(K5m)]
                except Exception:
                    kpm[fi4] = None; kpmE[fi4] = None
                sc4m = lr / max(N, 1) if self.opt == "gd" else lr        # optimizer step size (gd ⇒ η/N)
                for _ in range(max(1, ee)):                              # advance the frozen- & evolving-residual Jacobians ee GD steps
                    if torch.isfinite(fz["J"]).all() and float(fz["J"].norm()) < 1e8:   # cap ⇒ GN divergence stays finite (no eigh stall)
                        MrJm = torch.stack([hvp_S(model, fz["th0"], X, fz["r0"], fz["J"][k]) for k in range(M)])
                        fz["J"] = fz["J"] + sc4m * MrJm
                    if torch.isfinite(fz["Je"]).all() and torch.isfinite(fz["re"]).all() and torch.isfinite(fz["the"]).all() and float(fz["Je"].norm()) < 1e8:   # evolving-Qr
                        Yf = Y.reshape(-1)[:M]
                        if fz["reQ"] is None or fz["qsc"] % p4s == 0:                  # refresh the FIXED rQ=M_r(r̂Q,θ̂Q) at each s-step window boundary
                            fz["thQ"] = fz["the"].detach().clone(); fz["reQ"] = fz["re"].detach().clone()
                        fz["Je"] = fz["Je"] + sc4m * torch.stack([hvp_S(model, fz["thQ"], X, fz["reQ"].reshape(N, outD), fz["Je"][k]) for k in range(M)])   # Ĵ every step via the FIXED rQJ
                        g_e = precond_flow(self.model, self.opt, fz["Je"], fz["re"], N, lr)[0]   # P·(Ĵᵀr̂); gd ⇒ Ĵᵀr̂
                        _negm = float(g_e.norm())
                        if _negm > 1e4: g_e = g_e * (1e4 / _negm)                       # trust-region clamp
                        fz["the"] = fz["the"] + sc4m * g_e
                        dth_e = fz["the"] - fz["th0"]
                        HzD_e = jac_hvp(model, fz["th0"], X, dth_e)[:M]                  # Q0·Δθ̂
                        fz["re"] = Yf - (fz["f0"] + fz["J0e"] @ dth_e + 0.5 * (HzD_e @ dth_e))   # r̂ = Y − (f0 + J0Δθ̂ + ½Δθ̂ᵀQ0Δθ̂)
                        fz["qsc"] += 1
        if ka4m is not None:
            return {"kAct": ka4m, "off": [p4t0 - 10, p4t0 + 10, p4t0 + 20], "kPred": kpm, "kPredE": kpmE}
        return None


class TraceTracker:
    """Prediction-5 / trace (server s38): TRACE-STATISTIC prediction. At stat_init start four Jacobian
    trajectories {quad(T=0), cubic}×{live, self}: ΔJ=(η/N)Q_t[g]+½(η/N)²T[g,g], g=Jᵀr, Q_{t+1}=Q_t+(η/N)T[g]
    (cubic Eq-3 history); the SELF residual is the frozen QUADRATIC model r_q=Y−(f0+J0Δθ+½ΔθᵀQ0Δθ), the LIVE
    residual is the run's actual r. Each trajectory RE-ANCHORS at the current actual θ_t/J_t/residual every
    window (quad → qapprox, cubic → evcubic). Reports predicted Tr(NTK)=‖J‖²_F/N (WITH PSD) and its no-PSD
    accumulation (drop the per-step ‖ΔJ‖²), vs actual Tr(JᵀJ)/N and Tr(∇²L)/N (24 Hutchinson probes of the
    EXACT loss-Hessian hvp_L).

    MIRRORS server.py's g_trace per-step block (server.py:3777-3838) EXACTLY. `step()` returns the g_trace
    dict {t0, trGN, trHess, qLive, qSelf, cLive, cSelf} — each of the four a [withPSD, noPSD] pair.
    """

    def __init__(self, model, loss, N, outD, p, lr, ee, stat_init=0, qapprox=25, evcubic=25, opt="gd"):
        self.model = model; self.loss = loss
        self.N = N; self.outD = outD; self.p = p; self.M = N * outD
        self.lr = lr; self.ee = max(1, ee)
        self.stat_init = max(0, int(stat_init))
        self.qapprox = max(1, int(qapprox)); self.evcubic = max(1, int(evcubic)); self.opt = opt
        self.tr5 = None
        self.device = None; self.dtype = None       # captured lazily from th on the first step

    def _T2m5(self, th0, a, b, X):                   # T[a,b]=∇³f(θ0)[a,b] (M,p) at this trajectory's anchor θ0
        M = self.M; p = self.p
        an = float(a.norm())
        if an < 1e-30:
            return torch.zeros(M, p, dtype=self.dtype, device=self.device)
        ah = a / an
        return an * (jac_hvp(self.model, th0 + 1e-2 * ah, X, b)[:M]
                     - jac_hvp(self.model, th0 - 1e-2 * ah, X, b)[:M]) / (2e-2)

    def step(self, t, th, X, Y, Jc, rr):
        model = self.model; loss = self.loss
        N = self.N; M = self.M; p = self.p; lr = self.lr; ee = self.ee
        if Jc is None or rr is None:
            return None
        self.device = th.device; self.dtype = th.dtype
        Jm5 = Jc[:M]
        trGN = float((Jm5 * Jm5).sum()) / N            # actual Tr(JᵀJ)=‖J‖²_F ÷N — shown THROUGHOUT (not gated by stat_init)
        NP5 = 24                                        # actual Tr(∇²L) ÷N via Hutchinson with the EXACT loss-Hessian HVP
        thut = 0.0
        for kp in range(NP5):
            v = randn_vec(p, (0x7EAC5 + kp * 0x9E3779B1) & 0xFFFFFFFF, self.device, self.dtype).sign()
            thut += float(v @ hvp_L(model, loss, th, X, Y, v))
        trHess = thut / NP5
        if self.tr5 is None and t >= self.stat_init:    # START the forecast at stat_init; each trajectory RE-ANCHORS every window
            self.tr5 = {"traj": [{"cubic": c, "self": s_, "K": (self.evcubic if c else self.qapprox),
                                  "th0": None, "J0": None, "f0": None, "tr0": 0.0,
                                  "J": None, "HistG": [], "Dth": None, "trNo": 0.0, "sc": 1 << 30}   # sc huge ⇒ anchor on first tick
                                 for c in (False, True) for s_ in (False, True)]}
        if self.tr5 is None:                            # before stat_init: emit the ACTUAL traces only (no forecast yet)
            return {"t0": self.stat_init, "trGN": trGN, "trHess": trHess,
                    "qLive": None, "qSelf": None, "cLive": None, "cSelf": None}
        Yf5 = Y.reshape(-1)[:M]
        f_cur = Yf5 - rr[:M]                            # current ACTUAL output f(θ_t)
        tr_out = []
        for dtr in self.tr5["traj"]:
            if dtr["sc"] >= dtr["K"]:                    # (re-)anchor at the CURRENT ACTUAL θ_t/J_t/residual every K steps
                dtr["th0"] = th.detach().clone(); dtr["J0"] = Jm5.detach().clone(); dtr["f0"] = f_cur.detach().clone()
                dtr["tr0"] = float((Jm5 * Jm5).sum()); dtr["J"] = Jm5.detach().clone()
                dtr["Dth"] = torch.zeros(p, dtype=self.dtype, device=self.device)
                dtr["HistG"] = []; dtr["trNo"] = 0.0; dtr["sc"] = 0
            th0 = dtr["th0"]; J0 = dtr["J0"]; f0 = dtr["f0"]; Jc5 = dtr["J"]
            tr_out.append([float((Jc5 * Jc5).sum()) / N,             # predicted Tr(NTK) WITH PSD = ‖J‖²_F ÷N
                           (dtr["tr0"] + dtr["trNo"]) / N])          # ... WITHOUT PSD (drop the per-step ‖ΔJ‖²)
            for _ in range(ee):                                       # advance ee GD steps of the frozen-Q trajectory
                Jtr = dtr["J"]
                if not torch.isfinite(Jtr).all():                     # diverged ⇒ stop advancing this trajectory
                    break
                if dtr["self"]:
                    dth = dtr["Dth"]; HzD = jac_hvp(model, th0, X, dth)[:M]
                    r = Yf5 - (f0 + J0 @ dth + 0.5 * (HzD @ dth))    # quadratic-model self-residual
                else:
                    r = rr[:M]                                        # live-run residual
                g, scale = precond_flow(self.model, self.opt, Jtr, r, N, lr)   # (direction, step); gd ⇒ (Jᵀr, η/N) byte-identical
                Qg = jac_hvp(model, th0, X, g)[:M]
                if dtr["cubic"]:
                    for gk in dtr["HistG"]:
                        Qg = Qg + scale * self._T2m5(th0, gk, g, X)   # Q_t propagated (Eq-3 history)
                    dJ = scale * Qg + 0.5 * (scale ** 2) * self._T2m5(th0, g, g, X)   # ΔJ=scale·Q_t[g]+½·scale²·T[g,g]
                else:
                    dJ = scale * Qg                                   # quadratic: T=0, Q frozen ⇒ ΔJ=scale·Q0[g]
                dtr["trNo"] += 2.0 * float((Jtr * dJ).sum()); dtr["J"] = Jtr + dJ
                if dtr["cubic"]:
                    dtr["HistG"].append(g)
                if dtr["self"]:
                    dtr["Dth"] = dth + scale * g
                dtr["sc"] += 1
        return {"t0": self.stat_init, "trGN": trGN, "trHess": trHess,
                "qLive": tr_out[0], "qSelf": tr_out[1], "cLive": tr_out[2], "cSelf": tr_out[3]}


class Pred6Tracker:
    """Prediction-6 (server s39): FOUR INDEPENDENT adaptive-optimizer trajectories {GD, Signed-GD, Spectral(Muon),
    Gauss-Newton} from the SAME init θ0, each stepped with its OWN learning rate (lr6). Per tick it reports, for
    each trajectory: the top-50 Gauss-Newton eigenvalues (= the nonzero spectrum of JᵀJ = eigvals of the NTK JJᵀ)
    as a DESCENDING scree curve, the loss, and the sharpness λmax(∇²L) via bounded Lanczos.

    MIRRORS server.py's g_pred6 per-step block (server.py:4042-4075) EXACTLY. `step(t, th, X, Y)` returns the
    g_pred6 dict {t, gd/sign/spectral/gaussnewton:[≤50 eig], loss:{opt:..}, sharp:{opt:λmax∇²L}}. The live θ is
    used ONLY to seed all four trajectories at the first tick; thereafter each evolves on its own (NO momentum/WD).

    Parity: `spec6` (exact Jacobian eigenvalues) and `loss6` match the server to ~1e-12 fp64 / ~1e-4 fp32;
    `sharp6` carries the documented FD-vs-exact-HVP gap (~1e-3), because server.MlpModel.hvpL is finite-difference
    while eos_lab.hvp_L is exact autograd (identical to the §1 sharpness discrepancy)."""

    def __init__(self, model, loss, N, outD, p, lr6, ee, mV):
        self.model = model; self.loss = loss
        self.N = N; self.outD = outD; self.p = p; self.M = N * outD
        self.lr6 = dict(lr6); self.ee = max(1, ee); self.mV = mV
        self.p6 = None                                  # lazy-seeded {optimizer: θ}, all from the run's init θ0
        self.device = None; self.dtype = None

    def step(self, t, th, X, Y):
        model = self.model; loss = self.loss
        N = self.N; M = self.M; p = self.p; outD = self.outD
        self.device = th.device; self.dtype = th.dtype
        if self.p6 is None:                             # seed all 4 at the current θ (run init θ0)
            self.p6 = {k: th.detach().clone() for k in ("gd", "sign", "spectral", "gaussnewton")}
        _K6 = min(50, M)
        _mV6 = min(self.mV, 20)                         # bounded Lanczos for the per-trajectory sharpness
        _fin = lambda x: x if (x == x and -1e30 < x < 1e30) else None    # finite-or-None (JSON can't hold inf/nan)
        spec6 = {}; loss6 = {}; sharp6 = {}
        for okey in ("gd", "sign", "spectral", "gaussnewton"):
            thx = self.p6[okey]; olr = self.lr6[okey]
            Jx, _ox = jac_cols(model, thx, X); Jx = Jx[:M]
            lam6 = safe_eigvalsh(Jx @ Jx.t())                                # JJᵀ (NTK) eigenvalues, ascending; = nonzero Gauss-Newton spectrum
            spec6[okey] = [max(0.0, float(lam6[-1 - i])) for i in range(_K6)]  # top-50 (or M) DESCENDING (rank 1 = largest)
            if torch.isfinite(thx).all():                                    # loss + sharpness λmax(∇²L) at this θ (None once a trajectory diverges)
                loss6[okey] = _fin(float(loss.value(model.forward(thx, X), Y, N)))
                try:
                    sharp6[okey] = _fin(max(0.0, float(lanczos_extreme_vals(
                        lambda v: hvp_L(model, loss, thx, X, Y, v), p, 1, _mV6, 0x5EED1, self.device, self.dtype)[0][0])))
                except Exception:
                    sharp6[okey] = None
            else:
                loss6[okey] = None; sharp6[okey] = None
            for _ in range(max(1, self.ee)):                                 # advance ee steps per tick ⇒ stay in lockstep with the run's t-axis (eigevery>1 safe)
                if okey == "gaussnewton":                                    # GN needs the full J and residual r
                    Js, os = jac_cols(model, thx, X)
                    rx = (-N * loss.resid_cotangent(os.reshape(N, outD), Y, N)).reshape(-1)[:M]
                    thx = thx - olr * opt_dir(model, None, okey, Js[:M], rx)
                else:
                    thx = thx - olr * opt_dir(model, grad_loss(model, loss, thx, X, Y)[0], okey)
            self.p6[okey] = thx
        return {"t": t, **spec6, "loss": loss6, "sharp": sharp6}            # {t, gd/sign/spectral/gaussnewton:[≤50 eig], loss:{opt:..}, sharp:{opt:λmax∇²L}}


class RayTracker:
    """Prediction-4.2 (server s41): PHASE-2a RAY, MULTI-ANCHOR. Each tick, compute w1 = top eigenvector of
    M_r=Σ_k r_kQ_k (warm-started Lanczos on hvp_S) and cos(∇L, w1). RE-ANCHOR a fresh ray each time cos rises
    UP through prthr (rising edge) in a direction DIFFERENT from the previous anchor (|cos(w1,w1_prev)| < prdir):
    freeze θ_a, r_a, w1 and walk the straight ray θ(s)=θ_a+s·w1 (no training), predicting σ1(J)=√λmax(JJᵀ),
    ‖J‖_F, the step-clock t(s), and the D_J=D_r phase-2a exit. The grid span s_max is sized by a LOOKAHEAD of
    prK actual GD steps from θ_a so GD needs ~prK steps to cross it. GD-only (the +Jᵀr walk & η/N clock are GD).

    MIRRORS server.py's g_ray per-step block (server.py:4172-4233) EXACTLY (eos_lab primitives; note hvp_S's
    arg order (c, v) is reversed vs server hvpS(th,X,v,c)). The Lanczos WARM-STARTS from the previous tick's
    eigvec (self.ray_track_w1) via _lanczos_core(q0=...). `step()` returns the g_ray dict: anchor=True on a
    fresh anchor (grid + predicted curves + exit), anchor=False otherwise (the actual dot st/dt/sig1a/jfroa for
    the current anchor), or None (before ‖Jᵀr‖>0 / a pathological tick).

    PARITY: w1/cos and the D_J grid term go through hvp_S (M_r), so they carry the documented FD-vs-exact HVP
    gap on server's MLP (~1e-3); the pure-linear grid pieces (σ1, ‖J‖_F, D_r=‖JᵀJ∇L‖, the s_max lookahead, the
    clock) match server to ~1e-9. Warm-started 24-iter Lanczos ⇒ w1 tracks the server's eigvec sequence.
    """

    def __init__(self, model, loss, N, outD, p, lr, opt="gd", prthr=0.9, prm=40, prK=75.0, prdir=0.9):
        self.model = model; self.loss = loss
        self.N = N; self.outD = outD; self.p = p; self.M = N * outD
        self.lr = lr; self.opt = opt
        self.prthr = float(prthr); self.prm = max(4, int(prm))
        self.prK = max(1.0, float(prK)); self.prdir = min(0.999, float(prdir))
        # multi-anchor state (server: ray_prev_cos / ray_last_w1 / ray_cur / ray_nanch / ray_track_w1)
        self.ray_prev_cos = None; self.ray_last_w1 = None; self.ray_cur = None
        self.ray_nanch = 0; self.ray_track_w1 = None
        self.device = None; self.dtype = None

    def step(self, t, th, X, Y, Jc, rr):
        model = self.model; loss = self.loss
        N = self.N; M = self.M; p = self.p; lr = self.lr; outD = self.outD
        prthr = self.prthr; prm = self.prm; prK = self.prK; prdir = self.prdir
        if Jc is None or rr is None:
            return None
        self.device = th.device; self.dtype = th.dtype
        try:                                                         # a pathological tick skips the ray rather than aborting the run
            Jmr = Jc[:M]; rmr = rr[:M]
            _newanch = False; g_ray = None
            _Jrr = Jmr.t() @ rmr; _Jrrn = float(_Jrr.norm())         # ∇L ∝ −Jᵀr; watch cos(∇L, w1) EVERY tick to catch each new alignment event
            if _Jrrn > 1e-30:
                _q0r = self.ray_track_w1 if self.ray_track_w1 is not None else _randvec16(p, SEC21_SEED, self.device, self.dtype)   # WARM-START from the previous tick's eigvec; first tick uses the server's mulberry32 cold-start vector (SEC21_SEED)
                _Qr, _Tr, _kr = _lanczos_core(lambda v: hvp_S(model, th, X, rmr.reshape(N, outD), v),
                                              p, min(p, 24), 0, self.device, self.dtype, q0=_q0r)
                _mur, _Svr = safe_eigh(_Tr)
                _w1 = _Svr[:, int(torch.argmax(_mur))].to(device=self.device, dtype=self.dtype) @ torch.stack(_Qr)   # top M_r eigvec (unit, p)
                self.ray_track_w1 = _w1.detach().clone()             # seed next tick's Lanczos (sign-agnostic — Krylov is sign-invariant)
                _cos = abs(float(_w1 @ _Jrr)) / _Jrrn                # cos(∇L, w1) ∈ [0,1]
                if float(_w1 @ _Jrr) < 0:                            # canonicalize w1 to the DIRECTION OF MOTION (+Jᵀr)
                    _w1 = -_w1
                _rise = (self.ray_prev_cos is not None and self.ray_prev_cos < prthr and _cos >= prthr)   # rising edge UP through prthr
                _difdir = (self.ray_last_w1 is None) or (abs(float(_w1 @ self.ray_last_w1)) < prdir)       # direction different from the previous anchor
                if _rise and _difdir:                                # ★ NEW ALIGNMENT ANCHOR
                    _th0 = th.detach().clone(); _ra = rmr.detach().clone()
                    _sc = lr / max(N, 1) if self.opt == "gd" else lr
                    _thla = _th0.clone(); _smax = 1e-12              # LOOKAHEAD: prK actual GD steps → furthest reach along w1 = s_max (GD needs ~prK steps to cross)
                    for _ in range(int(round(prK))):
                        _thla = _thla - lr * grad_loss(model, loss, _thla, X, Y)[0]
                        _smax = max(_smax, float((_thla - _th0) @ _w1))
                    _cra = _ra.reshape(N, outD)
                    grs = []; gsig = []; gjf = []; gV = []; gDJ = []; gDr = []
                    for j in range(prm + 1):
                        sj = (_smax * j) / prm; thj = _th0 + sj * _w1
                        Jj = jac_cols(model, thj, X)[0][:M]
                        grs.append(sj); gjf.append(float(Jj.norm()))
                        gsig.append(float(safe_eigvalsh(Jj @ Jj.t())[-1].clamp_min(0.0).sqrt()))   # σ1(J)=√λmax(JJᵀ)
                        _Jtra = Jj.t() @ _ra; gV.append(max(_sc * float(_Jtra.norm()), 1e-30))     # per-STEP distance rate Δs/step
                        _gL = _Jtra / max(N, 1)                                                    # ∇L on the ray (r held = r_a)
                        gDr.append(float((Jj.t() @ (Jj @ _gL)).norm()))                            # D_r=‖J·ṙ‖=‖JᵀJ∇L‖
                        gDJ.append(float(hvp_S(model, thj, X, _cra, _gL).norm()))                  # D_J=‖J̇·r_a‖=‖M_r(s)∇L‖
                    gclock = [0.0]                                   # clock t(s)=∫ds′/V, trapezoid (frozen-r_a prediction)
                    for j in range(1, prm + 1):
                        gclock.append(gclock[-1] + (grs[j] - grs[j - 1]) * 0.5 * (1.0 / gV[j - 1] + 1.0 / gV[j]))
                    _ng = len(grs); sExit = None; tExit = None      # exit = first D_J−D_r sign change on the grid
                    for j in range(1, _ng):
                        _a = gDJ[j - 1] - gDr[j - 1]; _b = gDJ[j] - gDr[j]
                        if (_a < 0) != (_b < 0):
                            _fr = _a / (_a - _b) if (_a - _b) != 0 else 0.0
                            sExit = grs[j - 1] + _fr * (grs[j] - grs[j - 1]); tExit = gclock[j - 1] + _fr * (gclock[j] - gclock[j - 1]); break
                    self.ray_cur = {"idx": self.ray_nanch, "tstar": t, "th0": _th0, "ra": _ra, "w1": _w1.detach().clone()}   # ★ commit AFTER a successful grid
                    self.ray_last_w1 = _w1.detach().clone(); self.ray_nanch += 1
                    g_ray = {"anchor": True, "idx": self.ray_cur["idx"], "tstar": t, "s": grs, "sig1": gsig, "jfro": gjf,
                             "clock": gclock, "DJ": gDJ, "Dr": gDr, "sExit": sExit, "tExit": tExit,
                             "st": 0.0, "sig1a": gsig[0], "jfroa": gjf[0], "dt": 0}
                    _newanch = True
                self.ray_prev_cos = _cos
            if (not _newanch) and self.ray_cur is not None:          # ACTUAL point measured vs the CURRENT anchor (past anchors freeze)
                g_ray = {"anchor": False, "idx": self.ray_cur["idx"],
                         "st": float((th - self.ray_cur["th0"]) @ self.ray_cur["w1"]), "dt": t - self.ray_cur["tstar"],
                         "sig1a": float(safe_eigvalsh(Jmr @ Jmr.t())[-1].clamp_min(0.0).sqrt()), "jfroa": float(Jmr.norm())}
            return g_ray
        except Exception:
            return None


def run_predictions(cfg, device=None, dtype=None, cifar_dir=None):
    """Drive the prediction-widget trackers along a full-batch training run — the eos_lab fp64 reference for
    server.py's g_pred3 / g_pred3m / g_pred4 / g_pred4m / g_trace / g_pred6 SSE keys.

    Runs ONE optimizer trajectory from θ0 (the run's `optimizer`, mirroring server run_stream 4754-4759) and,
    every `eigevery`-th tick, steps the enabled trackers (s36→Pred-3, s37→Pred-4, s38→Trace(5.1), s39→Pred-6).
    Returns {"meta":..., "records":[{"t":t, "g_pred3":..., "g_pred3m":..., "g_pred4":..., "g_pred4m":...,
    "g_trace":..., "g_pred6":...}, ...]} (only the enabled keys per record). Full-batch only (predictions gate to
    MSE + small M); NOT part of run_job's diagnostics path.

    MIRRORS server.py run_stream's per-tick dispatch: Jc/rr are computed once per tick and shared across the
    residual-based trackers (server.py:3667-3670); Pred-6 recomputes its own jac_cols per trajectory."""
    from .train import resolve_device, effective_dims
    dev = resolve_device(device)
    if dtype is None:
        dtype = torch.float32 if dev.type == "cuda" else torch.float64
    torch.set_grad_enabled(False)

    P = cfg.P()
    dataset = cfg.dataset
    inD, outD = effective_dims(cfg)
    model = build_model(cfg.arch, inD, outD, P, dev, dtype)
    loss = build_loss(cfg.loss)
    if loss.name != "mse":
        raise ValueError("predictions need MSE loss (the residual / NTK theories are for squared loss).")
    p = model.p
    N = int(cfg.nsamp); M = N * outD
    lr = float(cfg.lr); ee = max(1, int(cfg.eigevery)); opt = cfg.optimizer
    th, X, Y, _, _ = init_data_theta(model, P, dataset, N, inD, outD, dev, dtype, cifar_dir)

    mV = min(p, max(2 * int(cfg.neig) + 16, 28))                 # Lanczos size (mirrors Diagnostics.mV / server mV)
    lr6 = {"gd": float(cfg.lr_gd), "sign": float(cfg.lr_sign),
           "spectral": float(cfg.lr_muon), "gaussnewton": float(cfg.lr_gn)}

    trk = {}
    if int(getattr(cfg, "s36", 0)):
        trk["pred3"] = Pred3Tracker(model, loss, N, outD, p, lr, ee, opt=opt,
                                    p3rep=int(cfg.p3rep), p3k=int(cfg.p3k))
    if int(getattr(cfg, "s37", 0)):
        trk["pred4"] = Pred4Tracker(model, loss, N, outD, p, lr, ee, p4t0=int(cfg.p4t0),
                                    p4s=int(cfg.p4s), p4rep=int(cfg.p4rep), opt=opt)
    if int(getattr(cfg, "s38", 0)):
        trk["trace"] = TraceTracker(model, loss, N, outD, p, lr, ee, stat_init=int(cfg.stat_init),
                                    qapprox=int(cfg.qapprox), evcubic=int(cfg.evcubic), opt=opt)
    if int(getattr(cfg, "s39", 0)):
        trk["pred6"] = Pred6Tracker(model, loss, N, outD, p, lr6, ee, mV)
    if int(getattr(cfg, "s41", 0)) and opt == "gd":                  # Pred-4.2 ray is GD-only (the +Jᵀr walk & η/N clock are GD-specific), mirroring server's `opt == "gd"` gate
        trk["ray"] = RayTracker(model, loss, N, outD, p, lr, opt=opt,
                                prthr=float(cfg.prthr), prm=int(cfg.prm), prK=float(cfg.prK), prdir=float(cfg.prdir))

    records = []
    for t in range(int(cfg.steps) + 1):
        if t % ee == 0:
            Jc, out_flat = jac_cols(model, th, X)
            rr = (-N * loss.resid_cotangent(out_flat.reshape(N, outD), Y, N)).reshape(-1)
            rec = {"t": t}
            if "pred3" in trk:
                rec["g_pred3"] = trk["pred3"].step(t, th, X, Y, Jc, rr)
                rec["g_pred3m"] = trk["pred3"].step_multi(t, th, X, Y, Jc, rr)
            if "pred4" in trk:
                rec["g_pred4"] = trk["pred4"].step(t, th, X, Y, Jc, rr)
                rec["g_pred4m"] = trk["pred4"].step_multi(t, th, X, Y, Jc, rr)
            if "trace" in trk:
                rec["g_trace"] = trk["trace"].step(t, th, X, Y, Jc, rr)
            if "pred6" in trk:
                rec["g_pred6"] = trk["pred6"].step(t, th, X, Y)
            if "ray" in trk:
                rec["g_ray"] = trk["ray"].step(t, th, X, Y, Jc, rr)
            records.append(rec)
        if t < int(cfg.steps):
            if opt == "gaussnewton":                              # GN needs the full J and residual r (mirror server run_stream 4754-4759)
                Jg, of = jac_cols(model, th, X)
                rrg = (-N * loss.resid_cotangent(of.reshape(N, outD), Y, N)).reshape(-1)
                th = th - lr * opt_dir(model, None, opt, Jg[:M], rrg[:M])
            else:
                th = th - lr * opt_dir(model, grad_loss(model, loss, th, X, Y)[0], opt)

    meta = {"p": p, "N": N, "M": M, "outD": outD, "steps": int(cfg.steps), "ee": ee,
            "lr": lr, "optimizer": opt, "device": str(dev), "dtype": str(dtype).replace("torch.", ""),
            "mV": mV, "lr6": lr6}
    return {"meta": meta, "records": records}
