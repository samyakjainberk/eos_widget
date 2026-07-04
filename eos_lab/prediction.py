"""
prediction — the EoS "prediction widget" forecasts (server.py's `/prediction` route), ported to eos_lab.

MIRRORS:  server.py  (_first_signchange / _max_abs_curvature / _pred_signchange / run_sweep, and the
                       per-step forecast blocks g_pred3 / g_pred3m / g_pred4 / g_pred4m / g_trace
                       inlined in run_stream's main loop)
          ↔ index_prediction.html  (the 4-plot forecast panel + the §1/§2 sweep scatter)

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
                     hvp_S, hvp_L, MlpModel)
from .data import init_data_theta
from .linalg import safe_eigvalsh, randn_vec


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
def pred_signchange(model, loss, th_start, X, Y, N, outD, p, lr, K, max_steps, cubic=True):
    """PREDICTED first sign-change of d = ‖J·ṙ‖−‖J̇·r‖ along the Eq-51 (§10) trajectory from th_start,
    with the quadratic model refrozen every K steps. Returns (t, censored). The residual is the frozen-quad
    self-residual r_q; jrd/jdr use the propagated J and r_q with Q frozen at the current anchor. Sign
    only ⇒ ∇L magnitude is irrelevant. MIRRORS server._pred_signchange exactly (matrix-free via eos_lab).
    cubic=True  ⇒ full §10 cubic: J += (η/N)·Qg + ½(η/N)²·T[g,g], Qg carries the 3rd-derivative T corrections.
    cubic=False ⇒ QUADRATIC approximation (prediction-1 plot 1): T=0 and Q FIXED ⇒ J += (η/N)·Q₀·g only."""
    dev, dt = th_start.device, th_start.dtype
    etaN = lr / max(N, 1)
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
        gL = -(1.0 / N) * (Jc.t() @ r_q)                  # predicted ∇L (MSE)
        jrd = float((Jc.t() @ (Jc @ gL)).norm())          # ‖J·ṙ‖ = ‖JᵀJ∇L‖
        jdr = float(hvp_S(model, th0, X, r_q.reshape(N, outD), gL).norm())   # ‖J̇·r‖ = ‖M_r∇L‖ (Q frozen at th0)
        d = jrd - jdr
        sg = (1 if d > 0 else (-1 if d < 0 else 0))
        if prev is not None and sg != 0 and sg != prev:
            return s, False
        if sg != 0:
            prev = sg
        g = Jc.t() @ r_q                                  # advance the self trajectory (Eq-51)
        Qg = jac_hvp(model, th0, X, g)[:M]
        if cubic:
            for gk in st["HistG"]:
                Qg = Qg + etaN * T2m(th0, gk, g)
            dJ = etaN * Qg + 0.5 * (etaN ** 2) * T2m(th0, g, g)   # full cubic (3rd-derivative T corrections)
            st["HistG"].append(g)
        else:
            dJ = etaN * Qg                                # QUADRATIC: T=0, Q fixed ⇒ J += (η/N)·Q₀·g
        st["J"] = Jc + dJ
        st["Dth"] = dth + etaN * g
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
      curv             — peak signed d²loss/dt² (max-|·| central 2nd difference of loss(t))
      swsumn, swsumn2  — the two summation windows echoed back
      diverged         — the pair blew up (loss NaN or >1e12) before `steps`

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
    for pi in range(npairs):
        lr = math.exp(rng.uniform(math.log(lrmin), math.log(lrmax)))
        std = math.exp(rng.uniform(math.log(stdmin), math.log(stdmax)))
        th = model.init_theta(seed0 + 1, std)
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
            gL = grad_loss(model, loss, th, X, Y)[0]
            jrd = float((Jm.t() @ (Jm @ gL)).norm())
            jdr = float(hvp_S(model, th, X, rr.reshape(N, outD), gL).norm())
            diffs.append(jrd - jdr); losses.append(lv)
            if t == Tstart:
                th_at_T = th.detach().clone()
            th = th - lr * gL
        tAct, censA = first_signchange(diffs, steps)
        if th_at_T is not None:                            # predict only if θ(Tstart) reached before divergence
            tPredRel, censP = pred_signchange(model, loss, th_at_T, X, Y, N, outD, p, float(lr), K, steps - Tstart, cubic=True)
            tPred = tPredRel + Tstart
            tPredQRel, censPQ = pred_signchange(model, loss, th_at_T, X, Y, N, outD, p, float(lr), K, steps - Tstart, cubic=False)   # prediction-1 plot 1: quadratic (T=0, Q fixed)
            tPredQ = tPredQRel + Tstart
        else:
            tPred, censP = steps, True; tPredQ, censPQ = steps, True
        alignNTK = float(align_acc / align_cnt) if align_cnt else 0.0   # residual↔top-3-NTK-eigvec alignment (tick colour for prediction-2/2b)
        curv = max_abs_curvature(losses)                   # peak signed d²loss/dt² over the finite prefix (legacy)
        sumD = float(sum(diffs[:swsumn]))                  # prediction-2: Σ d = Σ(jrd−jdr) over the first swsumn iterations (from 0)
        sumCurv = float(sum(curvature_at(losses, tt) for tt in range(min(swsumn, len(losses)))))   # prediction-2: Σ d²loss/dt² over the first swsumn iterations (from 0)
        sumD2 = float(sum(diffs[:swsumn2]))                # prediction-2b: Σ d over the first swsumn2 iterations (own window)
        nj2 = min(swsumn2, len(jnorms) - 1)                # prediction-2b: Σ ∇‖J‖ over the first swsumn2 iters = ‖J‖_{swsumn2} − ‖J‖_0 (telescoping)
        sumJgrad = float(sum(jnorms[i + 1] - jnorms[i] for i in range(nj2))) if nj2 >= 1 else 0.0
        yield {"i": pi, "lr": float(lr), "std": float(std),
               "tAct": tAct, "censA": censA or diverged, "tPred": tPred, "censP": censP,
               "tPredQ": tPredQ, "censPQ": censPQ,                                   # prediction-1 plot 1: quadratic (T=0, Q fixed) predicted t*
               "alignNTK": (alignNTK if alignNTK == alignNTK else 0.0),             # Σ|⟨r̂,v_k⟩| top-3 NTK — prediction-2/2b tick colour
               "sumD": (sumD if sumD == sumD else 0.0), "sumCurv": (sumCurv if sumCurv == sumCurv else 0.0),
               "sumD2": (sumD2 if sumD2 == sumD2 else 0.0), "sumJgrad": (sumJgrad if sumJgrad == sumJgrad else 0.0),
               "curv": (curv if curv == curv else 0.0), "swsumn": swsumn, "swsumn2": swsumn2, "diverged": diverged}


# ═══════════════════════════════════════════════════════════════════════ per-step forecast state machines
class Pred3Tracker:
    """Prediction-3 (server s36): after (jrd−jdr)=‖J·ṙ‖−‖J̇·r‖ first changes sign at t*, freeze J*=J(t*) and
    compare the ACTUAL residual with the frozen-J linear NTK theory r_{k+1}=(I−(η/N)J*J*ᵀ)r_k from r(t*)
    (1st order), plus a 2nd-order theory that adds ½(η/N)²(gᵀQ*g). Also drives the MULTI-FREEZE panel with
    (J,Q) frozen at t*−10 / t*+10 / t*+20 (with a back-fill of the earliest anchor).

    MIRRORS server.py's g_pred3 + g_pred3m per-step blocks (server.py:3631-3691) EXACTLY. Two records per
    step: `step()` returns g_pred3 (None until Jc/rr are available), `step_multi()` returns g_pred3m.

    The tracker owns the sign-change state (t*, frozen J*/θ*, the propagated theories), so t* found in `step`
    is visible to `step_multi` in the same tick — mirroring the server, where both blocks share p3_tstar.
    """

    def __init__(self, model, loss, N, outD, p, lr, ee):
        self.model = model; self.loss = loss
        self.N = N; self.outD = outD; self.p = p; self.M = N * outD
        self.lr = lr; self.ee = max(1, ee)
        # prediction-3 single-freeze state (server: p3_*)
        self.p3_prev_sign = None; self.p3_tstar = None
        self.p3_Jstar = None; self.p3_rtheory = None
        self.p3_thstar = None; self.p3_r2 = None
        # prediction-3 multi-freeze state (server: p3m)
        self.p3m = None
        # cached per-tick Jc/rr slices (set in step(), reused by step_multi())
        self._Jm3 = None; self._rm3 = None; self._th = None

    # -- server helper: one GD step of the frozen 1st- & 2nd-order theories (J,Q frozen at the anchor) --
    def _adv3m(self, Js, ths, r1, r2, X):
        N = self.N; lr = self.lr; M = self.M
        r1n = r1 - (lr / N) * (Js @ (Js.t() @ r1))                    # 1st: r=(I−(η/N)JJᵀ)r
        g2 = Js.t() @ r2
        quad2 = 0.5 * (lr / N) ** 2 * (jac_hvp(self.model, ths, X, g2)[:M] * g2).sum(dim=1)   # +½(η/N)²·(gᵀQ_k g)_k
        return r1n, r2 - (lr / N) * (Js @ (Js.t() @ r2)) + quad2

    def step(self, t, th, X, Y, Jc, rr):
        """g_pred3 record for this tick. Returns None when Jc/rr are unavailable (multi-sample gate off)."""
        model = self.model; loss = self.loss
        N = self.N; M = self.M; lr = self.lr; ee = self.ee
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
        if self.p3_tstar is None and self.p3_prev_sign is not None and sgn3 != 0 and sgn3 != self.p3_prev_sign:
            self.p3_tstar = t
            self.p3_Jstar = Jm3.detach().clone(); self.p3_rtheory = rm3.detach().clone()   # freeze J*, r_theory = r(t*)
            self.p3_thstar = th.detach().clone(); self.p3_r2 = rm3.detach().clone()         # freeze θ* (for Q*), r₂ = r(t*)
        if sgn3 != 0:
            self.p3_prev_sign = sgn3
        if self.p3_tstar is not None:
            na3 = float(rm3.norm()); nt3 = float(self.p3_rtheory.norm()); nt3b = float(self.p3_r2.norm())
            cos3 = (float(rm3 @ self.p3_rtheory) / (na3 * nt3)) if (na3 > 1e-30 and nt3 > 1e-30) else None
            cos3b = (float(rm3 @ self.p3_r2) / (na3 * nt3b)) if (na3 > 1e-30 and nt3b > 1e-30) else None
            g_pred3.update({"tstar": self.p3_tstar, "rAct": na3, "rThy": nt3, "cos": cos3,
                            "rThy2": nt3b, "cos2": cos3b})
            for _ in range(ee):   # advance the linear + 2nd-order theories ONE GD step per actual step
                self.p3_rtheory = self.p3_rtheory - (lr / N) * (self.p3_Jstar @ (self.p3_Jstar.t() @ self.p3_rtheory))
                g2 = self.p3_Jstar.t() @ self.p3_r2                                          # 2nd order: g=J*ᵀr₂
                quad2 = 0.5 * (lr / N) ** 2 * (jac_hvp(model, self.p3_thstar, X, g2)[:M] * g2).sum(dim=1)
                self.p3_r2 = self.p3_r2 - (lr / N) * (self.p3_Jstar @ (self.p3_Jstar.t() @ self.p3_r2)) + quad2
        return g_pred3

    def step_multi(self, t, th, X, Y, Jc, rr):
        """g_pred3m record (multi-freeze) for this tick. Call AFTER step() (it shares the frozen-J* state and
        the cached Jm3/rm3). Returns None until t* has occurred; matches server's g_pred3m emission."""
        N = self.N; M = self.M; lr = self.lr; ee = self.ee
        if Jc is None or rr is None:
            return None
        Jm3 = self._Jm3 if self._Jm3 is not None else Jc[:M]
        rm3 = self._rm3 if self._rm3 is not None else rr[:M]
        if self.p3m is None:
            self.p3m = {"buf": [], "th": [None, None, None]}       # rolling (t,J,r,‖r‖,θ) buffer + 3 frozen-(J,Q) theories
        p3m = self.p3m
        p3m["buf"].append((t, Jm3.detach().clone(), rm3.detach().clone(), float(rm3.norm()), th.detach().clone()))
        if len(p3m["buf"]) > 30:
            p3m["buf"] = p3m["buf"][-30:]
        offs = [-10, 10, 20]; backfill = None
        if self.p3_tstar is not None and p3m["th"][0] is None:      # theory-0 (t*−10): freeze from the buffer, BACK-FILL t*−10 … t
            tgt = self.p3_tstar - 10
            bi = min(range(len(p3m["buf"])), key=lambda i: abs(p3m["buf"][i][0] - tgt))
            J0s = p3m["buf"][bi][1]; th0s = p3m["buf"][bi][4]
            rth = p3m["buf"][bi][2].clone(); r2th = p3m["buf"][bi][2].clone(); bf = []
            for j in range(bi, len(p3m["buf"])):
                bf.append([p3m["buf"][j][0], p3m["buf"][j][3], float(rth.norm()), float(r2th.norm())])   # (step, ‖r_act‖, ‖r_1st‖, ‖r_2nd‖)
                nadv = (p3m["buf"][j + 1][0] - p3m["buf"][j][0]) if j + 1 < len(p3m["buf"]) else max(1, ee)   # buffer points are eigevery GD-steps apart → advance that many (not 1); last → ee to hand off to the ongoing advance
                for _ in range(max(1, nadv)):
                    rth, r2th = self._adv3m(J0s, th0s, rth, r2th, X)
            p3m["th"][0] = {"J": J0s, "th": th0s, "r": rth, "r2": r2th}; backfill = bf
        for k in (1, 2):                                            # theory-1/2 (t*+10, t*+20): freeze when the run reaches them
            if self.p3_tstar is not None and p3m["th"][k] is None and t >= self.p3_tstar + offs[k]:   # >= (not ==): eigevery>1 skips the exact offset tick
                p3m["th"][k] = {"J": Jm3.detach().clone(), "th": th.detach().clone(),
                                "r": rm3.detach().clone(), "r2": rm3.detach().clone()}
        thy = [None, None, None]; thy2 = [None, None, None]
        for k in range(3):
            if p3m["th"][k] is None or (k == 0 and backfill is not None):
                continue                                           # skip theory-0's single emit on the back-fill tick
            thy[k] = float(p3m["th"][k]["r"].norm()); thy2[k] = float(p3m["th"][k]["r2"].norm())
            for _ in range(ee):
                p3m["th"][k]["r"], p3m["th"][k]["r2"] = self._adv3m(
                    p3m["th"][k]["J"], p3m["th"][k]["th"], p3m["th"][k]["r"], p3m["th"][k]["r2"], X)
        if self.p3_tstar is not None:
            return {"rAct": float(rm3.norm()), "thy": thy, "thy2": thy2,
                    "off": offs, "tstar": self.p3_tstar, "backfill": backfill}
        return None


class Pred4Tracker:
    """Prediction-4 (server s37): frozen-M_r Jacobian propagation. At iteration t0 freeze θ0, J0=J(t0) and the
    residual r0; then propagate J_{s+1}=(I+(η/N)M_r)J_s with M_r=Σ_k r0_k Q_k(θ0) frozen (matrix-free via
    hvp_S), and compare the top-3 NTK eigenvalues λ(JJᵀ) of the predicted vs actual J. A second (EVOLVING-
    residual) variant propagates (Ĵ, r̂) coupled with r̂ re-anchored to the ACTUAL residual every s steps.
    Also drives the MULTI-FREEZE panel (anchors at t0−10 / t0+10 / t0+20).

    MIRRORS server.py's g_pred4 + g_pred4m per-step blocks (server.py:3693-3775) EXACTLY. `step()` returns
    g_pred4 (with kAct/kPred/kPredE + cos5/cos5E/cosGN5/cosGN5E), `step_multi()` returns g_pred4m.
    """

    def __init__(self, model, loss, N, outD, p, lr, ee, p4t0=10, p4s=10):
        self.model = model; self.loss = loss
        self.N = N; self.outD = outD; self.p = p; self.M = N * outD
        self.lr = lr; self.ee = max(1, ee)
        self.p4t0 = max(0, int(p4t0)); self.p4s = max(1, int(p4s))
        # single-freeze state (server: p4_*)
        self.p4_J = None; self.p4_th0 = None; self.p4_r0 = None
        self.p4_Je = None; self.p4_re = None; self.p4_re_next = None
        # multi-freeze state (server: p4m)
        self.p4m = None

    def step(self, t, th, X, Y, Jc, rr):
        model = self.model
        N = self.N; M = self.M; lr = self.lr; ee = self.ee
        p4t0 = self.p4t0; p4s = self.p4s
        if Jc is None or rr is None:
            return None
        Jm4 = Jc[:M]
        if self.p4_J is None and t >= p4t0:                        # freeze Q & residual at t0
            self.p4_J = Jm4.detach().clone(); self.p4_th0 = th.detach().clone()
            self.p4_r0 = rr[:M].reshape(N, self.outD).detach().clone()
            self.p4_Je = Jm4.detach().clone(); self.p4_re = rr[:M].detach().clone()
            self.p4_re_next = p4t0 + p4s                            # EVOLVING-residual: seed Ĵ, r̂, first re-anchor
        if self.p4_J is None:
            return None
        if self.p4_re_next is not None and t >= self.p4_re_next:    # re-anchor r̂ to the ACTUAL residual every s steps
            self.p4_re = rr[:M].detach().clone()
            while self.p4_re_next <= t:
                self.p4_re_next += p4s
        K4 = min(3, M); K5 = min(3, M)                             # top-3 everywhere
        try:                                                       # eigh → eigenvalues AND eigenvectors (top-3 cosines)
            Wa, Va = torch.linalg.eigh(Jm4 @ Jm4.t())              # actual NTK: eigvals asc, eigvecs = columns
            Wp, Vp = torch.linalg.eigh(self.p4_J @ self.p4_J.t())  # predicted NTK (frozen residual)
            We, Ve = torch.linalg.eigh(self.p4_Je @ self.p4_Je.t())  # predicted NTK (evolving residual)
            kAct4 = [max(0.0, float(Wa[-1 - i])) for i in range(K4)]
            kPred4 = [max(0.0, float(Wp[-1 - i])) for i in range(K4)]
            kPredE = [max(0.0, float(We[-1 - i])) for i in range(K4)]
            cos5 = [abs(float(Va[:, -1 - i] @ Vp[:, -1 - i])) for i in range(K5)]    # |cos| actual vs frozen-residual NTK eigvec
            cos5E = [abs(float(Va[:, -1 - i] @ Ve[:, -1 - i])) for i in range(K5)]   # |cos| actual vs evolving-residual NTK eigvec
            cosGN5 = []; cosGN5E = []                              # |cos| of the GN (JᵀJ) eigvecs = right-singular vecs z=Jᵀu/‖·‖
            for i in range(K5):
                za = Jm4.t() @ Va[:, -1 - i]; za = za / max(float(za.norm()), 1e-30)
                zp = self.p4_J.t() @ Vp[:, -1 - i]; zp = zp / max(float(zp.norm()), 1e-30)
                ze = self.p4_Je.t() @ Ve[:, -1 - i]; ze = ze / max(float(ze.norm()), 1e-30)
                cosGN5.append(abs(float(za @ zp))); cosGN5E.append(abs(float(za @ ze)))
        except Exception:                                          # fp32-GPU eigh hiccup ⇒ eigenvalues only
            ka = safe_eigvalsh(Jm4 @ Jm4.t()); kp = safe_eigvalsh(self.p4_J @ self.p4_J.t())
            kpe = safe_eigvalsh(self.p4_Je @ self.p4_Je.t())
            kAct4 = [max(0.0, float(ka[-1 - i])) for i in range(K4)]
            kPred4 = [max(0.0, float(kp[-1 - i])) for i in range(K4)]
            kPredE = [max(0.0, float(kpe[-1 - i])) for i in range(K4)]
            cos5 = []; cos5E = []; cosGN5 = []; cosGN5E = []
        g_pred4 = {"t0": p4t0, "s": p4s, "kAct": kAct4, "kPred": kPred4, "kPredE": kPredE,
                   "cos5": cos5, "cos5E": cos5E, "cosGN5": cosGN5, "cosGN5E": cosGN5E}
        for _ in range(ee):                                        # advance ONE GD step per actual step (ee GD steps between ticks)
            if torch.isfinite(self.p4_J).all():
                MrJ = torch.stack([hvp_S(model, self.p4_th0, X, self.p4_r0, self.p4_J[k]) for k in range(M)])   # frozen M_r·J
                self.p4_J = self.p4_J + (lr / max(N, 1)) * MrJ
            if torch.isfinite(self.p4_Je).all():                   # evolving-residual: coupled forward-Euler of (Ĵ, r̂), Q frozen at θ0
                MrJe = torch.stack([hvp_S(model, self.p4_th0, X, self.p4_re.reshape(N, self.outD), self.p4_Je[k]) for k in range(M)])
                self.p4_re = self.p4_re - (lr / max(N, 1)) * (self.p4_Je @ (self.p4_Je.t() @ self.p4_re))
                self.p4_Je = self.p4_Je + (lr / max(N, 1)) * MrJe
        return g_pred4

    def step_multi(self, t, th, X, Y, Jc, rr):
        model = self.model
        N = self.N; M = self.M; lr = self.lr; ee = self.ee
        p4t0 = self.p4t0; p4s = self.p4s
        if Jc is None or rr is None:
            return None
        Jm4m = Jc[:M]
        if self.p4m is None:
            self.p4m = [{"tf": p4t0 - 10, "J": None, "th0": None, "r0": None, "Je": None, "re": None, "ren": None},
                        {"tf": p4t0 + 10, "J": None, "th0": None, "r0": None, "Je": None, "re": None, "ren": None},
                        {"tf": p4t0 + 20, "J": None, "th0": None, "r0": None, "Je": None, "re": None, "ren": None}]
        K5m = min(3, M)                                            # top-3
        try:
            Wam = safe_eigvalsh(Jm4m @ Jm4m.t()); ka4m = [max(0.0, float(Wam[-1 - i])) for i in range(K5m)]
        except Exception:
            ka4m = None
        kpm = [None, None, None]; kpmE = [None, None, None]
        for fi4, fz in enumerate(self.p4m):
            if fz["tf"] >= 0 and t == fz["tf"] and fz["J"] is None:  # freeze M_r (θ, J, residual) at this offset iteration
                fz["J"] = Jm4m.detach().clone(); fz["th0"] = th.detach().clone()
                fz["r0"] = rr[:M].reshape(N, self.outD).detach().clone()
                fz["Je"] = Jm4m.detach().clone(); fz["re"] = rr[:M].detach().clone(); fz["ren"] = fz["tf"] + p4s
            if fz["J"] is not None:
                if fz["ren"] is not None and t >= fz["ren"]:        # re-anchor evolving residual to actual every s steps
                    fz["re"] = rr[:M].detach().clone()
                    while fz["ren"] <= t:
                        fz["ren"] += p4s
                try:
                    Wpm = safe_eigvalsh(fz["J"] @ fz["J"].t()); kpm[fi4] = [max(0.0, float(Wpm[-1 - i])) for i in range(K5m)]
                    Wpe = safe_eigvalsh(fz["Je"] @ fz["Je"].t()); kpmE[fi4] = [max(0.0, float(Wpe[-1 - i])) for i in range(K5m)]
                except Exception:
                    kpm[fi4] = None; kpmE[fi4] = None
                for _ in range(ee):                                # advance the frozen- & evolving-residual Jacobians ee GD steps
                    if torch.isfinite(fz["J"]).all():
                        MrJm = torch.stack([hvp_S(model, fz["th0"], X, fz["r0"], fz["J"][k]) for k in range(M)])
                        fz["J"] = fz["J"] + (lr / max(N, 1)) * MrJm
                    if torch.isfinite(fz["Je"]).all():
                        MrJmE = torch.stack([hvp_S(model, fz["th0"], X, fz["re"].reshape(N, self.outD), fz["Je"][k]) for k in range(M)])
                        fz["re"] = fz["re"] - (lr / max(N, 1)) * (fz["Je"] @ (fz["Je"].t() @ fz["re"]))
                        fz["Je"] = fz["Je"] + (lr / max(N, 1)) * MrJmE
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

    def __init__(self, model, loss, N, outD, p, lr, ee, stat_init=0, qapprox=25, evcubic=25):
        self.model = model; self.loss = loss
        self.N = N; self.outD = outD; self.p = p; self.M = N * outD
        self.lr = lr; self.ee = max(1, ee)
        self.stat_init = max(0, int(stat_init))
        self.qapprox = max(1, int(qapprox)); self.evcubic = max(1, int(evcubic))
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
        etaN5 = lr / max(N, 1)
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
                g = Jtr.t() @ r
                Qg = jac_hvp(model, th0, X, g)[:M]
                if dtr["cubic"]:
                    for gk in dtr["HistG"]:
                        Qg = Qg + etaN5 * self._T2m5(th0, gk, g, X)   # Q_t propagated (Eq-3 history)
                    dJ = etaN5 * Qg + 0.5 * (etaN5 ** 2) * self._T2m5(th0, g, g, X)   # ΔJ=(η/N)Q_t[g]+½(η/N)²T[g,g]
                else:
                    dJ = etaN5 * Qg                                    # quadratic: T=0, Q frozen ⇒ ΔJ=(η/N)Q0[g]
                dtr["trNo"] += 2.0 * float((Jtr * dJ).sum()); dtr["J"] = Jtr + dJ
                if dtr["cubic"]:
                    dtr["HistG"].append(g)
                if dtr["self"]:
                    dtr["Dth"] = dth + etaN5 * g
                dtr["sc"] += 1
        return {"t0": self.stat_init, "trGN": trGN, "trHess": trHess,
                "qLive": tr_out[0], "qSelf": tr_out[1], "cLive": tr_out[2], "cSelf": tr_out[3]}
