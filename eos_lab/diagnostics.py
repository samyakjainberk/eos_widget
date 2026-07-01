"""
Per-step diagnostics — the science of every panel, evaluated at one θ.

MIRRORS:  server.py's per-step block of `run_stream`  ↔  index.html sections §1–§9.
Section ↔ flag map (same as config.Config):
  §1  s1   loss, sharpness (= top eig of loss Hessian), residual, spectral edges of H
  §2  s2   top-n eigenvalues of  H, loss Hessian, Gauss–Newton G, residual term S
  §3  s3   bottom-n eigenvalues of the same four matrices
  §4  s4   J = ∇Σf projected onto top/bottom-n eigenvectors of H  (raw + Frobenius-normalized)
  §5  s5   SLQ spectral density of H, G, S, loss Hessian
  §6  s6   eigenspace rotation of H (principal angles of the ±-energy subspaces)
  §7  s7   NTK + function-Hessian tensor SVD                     (multi-sample)
  §8  s8   vec(J) onto right-singular vecs of the p×(p·dₙ) FH reshape   (multi-sample)
  §4b s9   J·r onto eigvecs of Q[u₁]; alignment with GN top eigvec g₁    (multi-sample)
  §4c s10  Q[u₁]·(J·r) onto GN eigvecs g_k                       (multi-sample)
  §4d s11  per-residual-sign-group J / J·r onto the shared H and Q[u₁] eigvecs (multi-sample)
  §9  s12  theoretical (Eq-13/21/22/23/29) vs empirical σ₁ over frozen-Q windows

The multi-sample sections (§7/§8/§4b/§4c/§4d and the multi branch of §9) form the M×p Jacobian Jc
explicitly, so they are gated by a size budget (`multi_ok`) and silently skipped when M or M·p is
too large — the run logs which sections were skipped.
"""
import math
import torch

from .models import (grad_sum_f, hvp_F, hvp_L, hvp_S, hvp_G, hvp_G2,
                     jac_cols, jac_hvp, grad_out, mlp_forward, grad_loss, grad_weighted, FD_EPS)
from .linalg import (lanczos_extreme, lanczos_extreme_vals, slq_density, hutch_trace, sym_eig_desc,
                     sign_to, pin_sign, pos_subspace, neg_subspace, principal_angles, randn_vec,
                     sec12_payload, sec13_stats, SEC13_KS, SEC13_DOMS, sec14_payload, SEC12_KFULL,
                     batched_lanczos_extreme, sec15_stats, sec15_panel34, sec15_panelB,
                     safe_eigh, safe_eigvalsh)
from .rng import mulberry32

# §11 render budget: emit only ≤this many points per grid above this size (mirror server.G3D_MAXPTS).
G3D_MAXPTS = 10000


def _g3d_pack(flat, K=G3D_MAXPTS):
    """≤K points spanning the value range → (values, indices | None). Half top-|value| (structure) + half a
    random sample of the near-zero bulk, so colours show a fine white→saturated gradient (mirror server)."""
    n = int(flat.numel())
    if n <= K:
        return flat.detach().cpu().tolist(), None
    ktop = K // 2
    top = torch.topk(flat.abs(), ktop).indices
    rnd = torch.randint(0, n, (K - ktop,), device=flat.device)
    idx = torch.cat([top, rnd])
    return flat[idx].detach().cpu().tolist(), idx.to(torch.int64).cpu().tolist()

# Fixed Lanczos / probe seeds (match server.run_stream so curves line up closely).
SEED_H, SEED_L, SEED_G, SEED_S = 0xA53F9, 0x5EED1, 0x6A17C, 0x7B23D
SEED_SLQ_H, SEED_SLQ_G, SEED_SLQ_S, SEED_SLQ_L = 0x11, 0x22, 0x33, 0x44


def fh_frozen(model, theta, X, nprobe, n7, nM, mV, M, device, dtype):
    """§9 frozen function-Hessian-tensor SVD at θ. Returns {gamma[i], y[i]∈R^M, z[i][j]∈R^p, tau[i][j]}.
    MIRRORS server.fh_frozen."""
    Jc, _ = jac_cols(model, theta, X)
    p = Jc.shape[1]
    GH = torch.zeros(M, M, dtype=dtype, device=device)
    for pr in range(nprobe):
        Hz = jac_hvp(model, theta, X, randn_vec(p, (0xC0FFEE + pr * 40503) & 0xFFFFFFFF, device, dtype))
        GH += Hz @ Hz.t()
    GH /= nprobe
    Hv, Vh = sym_eig_desc(GH)
    N = X.shape[0]
    out = {"gamma": [], "y": [], "z": [], "tau": []}
    for i in range(min(n7, M)):
        out["gamma"].append(math.sqrt(max(float(Hv[i]), 1e-30)))
        out["y"].append(Vh[:, i])
        tv, bv, tval, bval = lanczos_extreme(
            lambda v: hvp_S(model, theta, X, Vh[:, i].reshape(N, -1), v), p, nM, mV,
            (0x9E37 + i * 0x1000) & 0xFFFFFFFF, device, dtype)
        out["z"].append(list(tv) + list(bv))
        out["tau"].append(list(tval) + list(bval))
    return out


class Diagnostics:
    def __init__(self, model, loss, cfg, device, dtype, max_M=2048, max_Mp=700_000_000):
        self.model, self.loss, self.device, self.dtype = model, loss, device, dtype
        P = cfg.P()
        self.dataset = P.get("dataset", "synthetic")   # owt is a token-LM (3D out) → no scalar residual y−f
        self.s1, self.s2, self.s3, self.s4 = P["s1"], P["s2"], P["s3"], P["s4"]
        self.s5, self.s6, self.gs = P["s5"], P["s6"], P["gs"]
        self.s7, self.s8, self.s9, self.s10, self.s11, self.s12 = (
            P["s7"], P["s8"], P["s9"], P["s10"], P["s11"], P["s12"])
        self.s13 = P.get("s13", 0)          # §7a NTK alignment (residual→NTK, NTK→FH-SVD)
        self.s14 = P.get("s14", 0)          # §9c: σ₁ predictions vs the FULL loss-Hessian sharpness λmax(∇²L)
        self.s15 = P.get("s15", 0)          # §9d: predictions with the residual self-computed by the quadratic model
        self.s16 = P.get("s16", 0)          # §9d-c: §9d predictions vs the full loss-Hessian sharpness
        self.s17 = P.get("s17", 0)          # §10: CUBIC approximation (Eq-47/51 σ₁ predictions, exact J&Q propagation)
        self.s18 = P.get("s18", 0)          # §11: 3D grids T1=JᵢᵀQⱼJₖ, T2=uⱼuₖT1, T3=rᵢuⱼuₖT1 (multi-sample, small M)
        self.s19 = P.get("s19", 0)          # §12: per-sample Q_i eigenvector cross-similarity (multi-sample, small N & p)
        self.s20 = P.get("s20", 0)          # §14: Tr(ΔNTK) per-triplet decomposition cubes (shares §12 eigenpairs)
        self.s21 = P.get("s21", 0)          # §13: residual-weighted curvature G1/G2/G3 vs exact ref (own toggle; shares §12 Lanczos)
        self.s22 = P.get("s22", 0)          # §12b: per-sample projection panels (s19=§12a angles+diag-align; both share the §12 Lanczos)
        self.s23 = P.get("s23", 0)          # §15: 2nd-difference decomposition of ‖J‖²_F & σ₁ (own toggle; holds 3 eig-ticks)
        # §18-§25 per-step sections (server flags s26..s33; §22/§23 are surrogate REPLACE drivers run in train.py).
        self.s26 = P.get("s26", 0)          # §18: per-sample (sample1 vs sample2) projections — re-slice of the §12 proj arrays (N=2 render)
        self.s27 = P.get("s27", 0)          # §19: one-step grad-norm change A_t + tr(NTK) (MSE+GD only)
        self.s28 = P.get("s28", 0)          # §20: M_r=Σr_kQ_k spectral histograms (cube cadence)
        self.s28k = max(1, P.get("s28k", 40))   # §20: # eigenpairs per side (top-K ⊕ bottom-K) of M_r
        self.s29 = P.get("s29", 0)          # §21: residual↔spectrum alignment (NTK + M_r + GN panels; cube cadence)
        self.s29k = max(1, P.get("s29k", 40))   # §21: # NTK eigvecs (top) & M_r eigenpairs per side
        self.s32 = P.get("s32", 0)          # §24: A=JJᵀr & B=(η/2N)JrᵀQ_kJr alignment vs r and top-4 NTK eigvecs
        self.s33 = P.get("s33", 0)          # §25: ‖∇L‖ + d/dt(J·r) split + 5-gradient cosine alignments (needs 3 consecutive steps)
        self.s34 = P.get("s34", 0)          # §26: eigenvector-direction drift |cos(v_i(t),v_i(t−k))| of top-3 GN/NTK & top-3⊕bottom-3 M_r
        self.divreg = float(P.get("divreg", 0.3))   # §15 divergence ε: softens the 0/0 spike at D²→0 inflections (0 = exact /D²)
        self._sec15_hist = []               # §15: rolling buffer of the last 2 eig-ticks' {th, J, r} (need t−1 and t−2)
        self._sec25_hist = []               # §25: rolling buffer of the last 2 ticks' {th, J, r, t} (II/III need t−1 and t−2)
        self._sec25_rhist = []              # §25: rolling buffer of the last 101 ticks' {t, r} for cos(r_t, r_{t−k}), k∈{10,20,30,50,100}
        self._sec26_hist = []               # §26: rolling buffer of the last 101 ticks' {t, gn/ntk/mrTop/mrBot eigvecs} for eigenvector-direction drift
        self._sec14_rhist = []              # §14: per-sample residual history (last ≤5 ticks) for the eq-③ ratio
        self._sec14_prev = None             # §14: previous eig-tick's {TV,TW,BV,BW,r,Jg} (u,σ,r_j,J_k at t; cur gives J_i,r_k at t+1)
        self._sec13_prev = None             # §13: previous eig-tick's per-sample {TV,TW,BV,BW,r,Jg} (Q_i,Q_k & J',r at t-1)
        self._sec13_qjprev = None           # §13 panel 4: previous eig-tick's {QJ=(N_j,N_i,p) Q_iJ_j, r} for the ‖A−B‖/‖A‖ asymmetry
        if loss.name == "ce":
            # CE uses the generic residual r = −∂L/∂z = softmax(z)−onehot (the loss-output cotangent). §4
            # (J→H) and §6 (rotation) are model-only; §7/§7a/§8/§4b/§4c use the function NTK (Jᵤ·Jᵤᵀ) plus
            # this residual, so they're all valid for CE — and gated by the same `multi_ok` size budget
            # (feasible for e.g. CIFAR-CE, M=N·classes; skipped at LM vocab sizes). Off for CE: §4d (its
            # sign-groups need a scalar residual) and the whole §9 family (§9/§9b/§9c/§9d/§9d-c — the
            # Eq-13/21/22/23/29 σ₁ recursion is derived for squared loss only).
            self.s11 = self.s12 = self.s14 = self.s15 = self.s16 = self.s17 = False
        p = model.p
        half = max(1, p // 2)
        self.p = p
        self.N = cfg.nsamp
        self.outD = model.oc
        self.M = self.N * self.outD
        self.multi = self.M > 1
        # multi-sample sections form the M×p Jacobian — gate by a size budget
        self.multi_ok = self.multi and self.M <= max_M and self.M * p <= max_Mp
        # §12a/§12b also run for a SINGLE sample (N=1): per-sample proj needs only one Q_i=∇²f_i (pair-based angle/
        # cross panels stay empty). §13/§14 still require multi (N³ cubes). MIRRORS server. (Lets §12b be checked vs §4.)
        self.s12single = bool(self.s19 or self.s22) and not self.multi and self.M * p <= max_Mp
        self.n = min(max(1, cfg.neig), half)
        self.kth = min(max(1, cfg.kth), half)
        self.nSub = min(max(self.n, self.kth, (10 if self.s6 else 1)), half)
        self.n7 = min(self.n, max(self.M, 1))
        self.n8 = min(self.n, max(1, p // 2))
        self.nResid = self.M             # §1 residual plot: ALL samples' residual evolution (was capped at 12)
        self.efrac = min(1.0, max(0.5, cfg.energyp / 100.0))
        self.nprobe = max(1, cfg.slqprobes)
        self.lr = cfg.lr
        self.optimizer = getattr(cfg, "optimizer", "gd")   # §19 (s27) is GD-only (the A_t formula uses the GD update)
        self.thr = 2.0 / cfg.lr                      # edge-of-stability threshold 2/η
        self.eigevery = max(1, cfg.eigevery)
        # Throttle the heaviest, slowly-varying panels so a run stays responsive (mirrors server.run_stream):
        #   §5 SLQ → ~50 density snapshots over the whole run (slqStride);  §7 FH-eigvec projections + §8
        #   FH-reshape SVD → every `heavyevery`-th tick.  The cheap core (§1/§2/§3/§4/§6 eigenvalues, §7a
        #   alignment, §9 theory) stays every tick.  Without this eos_lab recomputed §5 (4 SLQ densities)
        #   and §8 (hvpG2 Lanczos) every step — by far the dominant per-step cost.
        self.heavyevery = max(1, getattr(cfg, "heavyevery", 4))
        self.slqStride = max(1, math.ceil((cfg.steps // self.eigevery + 1) / 50))
        self.qapprox = max(1, cfg.qapprox)
        self.cubicapprox = max(1, getattr(cfg, "cubicapprox", 10))   # §10 cubic-window length
        self.grid3dcap = max(1, getattr(cfg, "grid3dcap", 30))       # §11 3D-grid size cap (skip when M exceeds it)
        self.sec12ncap = max(1, getattr(cfg, "sec12ncap", 500))      # §12 sample cap (gates N; no p-cap — large p is slow)
        self.cubeevery = max(1, getattr(cfg, "cubeevery", 1))        # §11-§14 cube snapshot cadence (1=every tick; mirrors server g3dstride)
        self._cube_n = 0                                             # diagnostic-tick counter for the cube cadence
        self.qmode = cfg.qmode
        self.tset = cfg.tset
        # Krylov depths (match server.run_stream)
        self.mF = min(p, max(2 * self.nSub + 24, 44))
        self.mV = min(p, max(2 * self.n + 16, 28))
        self.mSLQ = min(p, 40)
        # cross-step sign-tracking state
        self._prev_top = [None] * self.nSub
        self._prev_bot = [None] * self.nSub
        self._prev_pos = None
        self._prev_neg = None
        self._prev_ntk = [None] * self.n7
        self._prev_uj = [None] * self.n7
        self._prev_vh = [None] * self.n7
        # §7a Δλ·Δr running products (synchronous + 1-step lagged), with running time-averages
        self._prev_sharp7a = None; self._prev_ntkR7a = None; self._prev_prev_ntkR7a = None
        self._sumGsync7a = [0.0] * self.n7; self._sumGlag7a = [0.0] * self.n7
        self._cntGs7a = 0; self._cntGl7a = 0
        self._prev_m1 = [None] * 4
        self._prev_m2 = [None] * 4
        self._prev_g2 = [None] * self.n8
        self._prev_q9t = [None] * self.n
        self._prev_q9b = [None] * self.n
        # §9 frozen-window state
        self._thT0 = -1
        self._thTh0 = None
        self._thBase = 0.0
        self._thJp = None
        self._thJ = None
        self._thFroz = None
        self._thAcc1 = 0.0
        self._thAcc2 = 0.0
        self._thProd3 = 1.0
        self._thProd4 = 1.0
        self._thProd5 = 1.0
        self._thAccPSD = 0.0     # §9b: accumulated 2nd-order PSD term ‖ΔJᵀu₁‖² (≥0), multi
        self._thAccPSD1 = 0.0    # §9b: accumulated PSD term ‖ΔJ‖² (≥0), single-sample
        # §9d: identical predictions, but the residual is SELF-COMPUTED by the frozen quadratic model
        # (closed loop). f₀,J₀ frozen at window start; Δθ is the quad-GD displacement θ_quad−θ₀ from θ₀.
        self._thJ0 = None        # frozen J₀ (M,p) — linear term of f_quad
        self._thF0 = None        # frozen f₀ = Y−r (M,) — constant term of f_quad
        self._thDth = None        # Δθ (p,) — quad-GD displacement within the window (multi)
        self._thJ_d = None        # propagated quad Jacobian J₀+Q[Δθ]
        self._thProd3_d = self._thProd4_d = self._thProd5_d = 1.0
        self._thAcc2_d = 0.0; self._thAccPSD_d = 0.0
        self._thJp0 = None; self._thF0s = 0.0; self._thDth_s = None   # single-sample §9d
        self._thJp_d = None; self._thAcc1_d = 0.0; self._thAccPSD1_d = 0.0
        # §10 cubic-approximation window state. T (3rd derivative) is frozen at θ₀ and accessed matrix-free as
        # finite differences of the HVPs; BOTH J and Q are propagated (Q via the (r_k, J_k/g_k) history, so a
        # window costs O(window²) HVPs). `_cn` = number of probe vectors for the Hutchinson ‖ΔQ‖/‖Q₀‖ estimate.
        self._cT0 = -1; self._cTh0 = None; self._cBase = 0.0
        self._cJ0 = None             # frozen J at window start (single: (p,); multi: (M,p)) — for ‖ΔJ‖/‖J₀‖
        self._cn = max(1, min(self.nprobe, 4))
        self._cZ = None              # fixed Hutchinson probes z_i (list of (p,)) for ‖Q‖_F; reused across window
        self._cQ0z = None            # Q₀·z_i (list; single (p,), multi (M,p)) — denominator ‖Q₀‖_F
        self._cdQz = None            # running ΔQ·z_i over the window (same shapes)  — numerator ‖ΔQ‖_F
        # single-sample (Eq-47): propagated J (p,), Q-history (r_k, J_k), Eq-47 accumulators (±PSD)
        self._cJp = None; self._cHistR = None; self._cHistJ = None
        self._cAcc47 = 0.0; self._cAccPSD47 = 0.0
        # multi-sample (Eq-51): cubic-propagated J (M,p) + its g-history; accumulators for Eq-51 (±PSD) and for the
        # same cubic trajectory WITHOUT the η² (½T[Δθ,Δθ]) term of ΔJ (±PSD)
        self._cJ = None; self._cHistG = None
        self._cAcc51 = 0.0; self._cAccPSD51 = 0.0
        self._cAcc51n = 0.0; self._cAccPSD51n = 0.0

    def sections_skipped(self):
        """Names of enabled sections that are auto-skipped because the M×p Jacobian is too large."""
        if self.multi_ok or not self.multi:
            return []
        flags = {"s7": self.s7, "s8": self.s8, "s9": self.s9, "s10": self.s10, "s11": self.s11,
                 "s13": self.s13}
        return [k for k, on in flags.items() if on]

    # -- hvp closures bound to the current θ -----------------------------------------------------
    def _hF(self, th, X):
        return lambda v: hvp_F(self.model, th, X, v)

    def _hL(self, th, X, Y):
        return lambda v: hvp_L(self.model, self.loss, th, X, Y, v)

    def _hG(self, th, X):
        return lambda v: hvp_G(self.model, self.loss, th, X, v)

    def _hS(self, th, X, c):
        return lambda v: hvp_S(self.model, th, X, c, v)

    def compute(self, theta, X, Y, t, pos_rows=None, neg_rows=None):
        """Return a dict of all enabled diagnostics at step t. pos_rows/neg_rows feed §4d."""
        cube_emit = (self._cube_n % self.cubeevery == 0)   # §11-§14 cube cadence (mirrors server eigTick % g3dstride)
        self._cube_n += 1
        th = theta
        N, outD, M, p = self.N, self.outD, self.M, self.p
        n, nSub, n7, n8 = self.n, self.nSub, self.n7, self.n8
        dev, dt = self.device, self.dtype
        lr = self.lr

        J, out = grad_sum_f(self.model, th, X)              # J = ∇Σf (p,)
        loss_val = float(self.loss.value(out, Y, N))
        cS = self.loss.resid_cotangent(out, Y, N)           # ∂L/∂out (feeds the S term + SLQ-S)
        rec = {"t": t, "loss": loss_val, "thr": self.thr}
        # cadence gates for the heavy panels (see __init__): heavy_tick → §7 projections + §8; slq_tick → §5.
        tick = t // self.eigevery
        heavy_tick = (tick % self.heavyevery == 0)
        slq_tick = (tick % self.slqStride == 0)
        if self.loss.name != "ce" and self.dataset != "owt":  # CE, and the OWT LM under ANY loss, have no
            r = (Y - out).reshape(-1)                          #   scalar residual y−f (Y are token ids, shape ≠ out)
            rec["resid_mean"] = float(r.mean())
            rec["resid_rms"] = float(r.pow(2).mean().sqrt())
            rec["resid_head"] = r[:self.nResid].detach().cpu().tolist()

        need_vecs = self.s4 or self.s6 or self.s11    # §4d also projects onto the H eigvecs (match the browser)
        need_vals = self.s1 or self.s2 or self.s3 or need_vecs

        # ---- function Hessian H (values, and vectors if §4/§6) ----
        feTop = feBot = []
        feTV = feBV = None
        if need_vecs:
            feTV, feBV, feTop, feBot = lanczos_extreme(self._hF(th, X), p, nSub, self.mF, SEED_H, dev, dt)
        elif need_vals:
            feTop, feBot = lanczos_extreme_vals(self._hF(th, X), p, n, self.mV, SEED_H, dev, dt)
        if feTop:
            # H = ∇²(Σf) is summed over the M=N·oc outputs (out.sum()), so its eigenvalues grow ∝N.
            # The loss Hessian / sharpness average that M-sum by 1/N, so divide H eigenvalues by N to
            # put them on the same per-sample scale (the *eigenvectors* feTV/feBV are scale-free and
            # stay raw — §4 projections and §6 subspaces are unaffected).
            rec["H_top"] = [v / N for v in feTop[:n]]
            rec["H_bot"] = [v / N for v in feBot[:n]]
            rec["H_edge_max"], rec["H_edge_min"] = feTop[0] / N, feBot[0] / N

        # ---- loss Hessian (sharpness = top eigenvalue) ----
        if self.s1 or self.s2 or self.s3:
            hlt, hlb = lanczos_extreme_vals(self._hL(th, X, Y), p, n, self.mV, SEED_L, dev, dt)
            rec["lossH_top"], rec["lossH_bot"] = hlt, hlb
            rec["sharpness"] = hlt[0]

        # ---- Gauss–Newton G and residual term S (loss Hessian = G + S) ----
        if (self.s2 or self.s3) and self.gs:
            gnt, gnb = lanczos_extreme_vals(self._hG(th, X), p, n, self.mV, SEED_G, dev, dt)
            srt, srb = lanczos_extreme_vals(self._hS(th, X, cS), p, n, self.mV, SEED_S, dev, dt)
            rec["G_top"], rec["G_bot"] = gnt, gnb
            rec["S_top"], rec["S_bot"] = srt, srb

        # sign-track H eigvecs once (shared by §4, §4d, §6)
        if need_vecs and feTV is not None:
            for i in range(nSub):
                feTV[i] = sign_to(feTV[i], self._prev_top[i]); self._prev_top[i] = feTV[i]
                feBV[i] = sign_to(feBV[i], self._prev_bot[i]); self._prev_bot[i] = feBV[i]

        # ---- §4 |⟨J,u⟩| onto top-/bottom-n eigvecs u of H=Σ_aQ_a (3 phases); σ = the H eigenvalue (feTop/feBot, signed) ----
        if self.s4 and feTV is not None:
            jn = max(float(J.norm()), 1e-30)
            aT = [abs(float(J @ feTV[i])) for i in range(n)]       # |⟨J,u⟩| onto top-i eigvec
            aB = [abs(float(J @ feBV[i])) for i in range(n)]
            sigT = [float(feTop[i]) / N for i in range(n)]         # σ ÷ N (per-sample scale, like §2)
            sigB = [float(feBot[i]) / N for i in range(n)]
            rec["jtN"] = [x / jn for x in aT]                      # phase 1 (alignment): |⟨J,u⟩|/‖J‖
            rec["jbN"] = [x / jn for x in aB]
            rec["jt"] = [aT[i] * sigT[i] for i in range(n)]        # phase 2 (power iteration): |⟨J,u⟩|·σ
            rec["jb"] = [aB[i] * sigB[i] for i in range(n)]
            if self.loss.name != "ce" and self.dataset != "owt":  # phase 3 (residual dominance): J·r=Σ_a r_a∇f_a (cf §4b). Skipped for CE/owt.
                if out.numel() == 1:                              # single scalar residual ⇒ J·r=r·J: pull r OUTSIDE |·| (signed)
                    r0 = float((Y - out).reshape(-1)[0])
                    rec["jrt"] = [r0 * aT[i] * sigT[i] for i in range(n)]
                    rec["jrb"] = [r0 * aB[i] * sigB[i] for i in range(n)]
                else:                                             # multi: contract inside the |·| — |⟨J·r,u⟩|·σ
                    Jr = grad_weighted(self.model, th, X, Y - out)
                    rec["jrt"] = [abs(float(Jr @ feTV[i])) * sigT[i] for i in range(n)]
                    rec["jrb"] = [abs(float(Jr @ feBV[i])) * sigB[i] for i in range(n)]

        # ---- §6 eigenspace rotation ----
        if self.s6 and feTV is not None:
            posS = pos_subspace(feTop, feTV, self.efrac)
            negS = neg_subspace(feBot, feBV, self.efrac)
            rec["rot_pos"] = principal_angles(self._prev_pos, posS) if self._prev_pos is not None else None
            rec["rot_neg"] = principal_angles(self._prev_neg, negS) if self._prev_neg is not None else None
            rec["dim_pos"], rec["dim_neg"] = len(posS), len(negS)
            self._prev_pos, self._prev_neg = posS, negS

        # ---- shared multi-sample Jacobian columns Jc (M,p) + residual rr (M,) ----
        Jc = rr = None
        want_multi = self.multi_ok and (self.s7 or self.s8 or self.s9 or self.s10 or self.s11
                                         or self.s12 or self.s13 or self.s15 or self.s16 or self.s17
                                         or self.s18 or self.s19 or self.s20 or self.s21 or self.s22 or self.s23
                                         or self.s26 or self.s27 or self.s28 or self.s29 or self.s32 or self.s33 or self.s34)
        # §15/§19/§20/§21/§24/§25/§26 (s23/s27/s28/s29/s32/s33/s34) ALSO run for a single sample (N=1) — they only
        # need the M×p Jacobian (M=N·outD), gated by grid3dcap, not the full `multi_ok` budget. MIRRORS server.py:3210.
        single_sample = ((self.s23 or self.s27 or self.s28 or self.s29 or self.s32 or self.s33 or self.s34)
                         and N <= self.grid3dcap)
        if want_multi or self.s12single or single_sample:
            Jc, out_flat = jac_cols(self.model, th, X)
            rr = (-N * cS).reshape(-1)        # generic residual −N·∂L/∂out: Y−f (MSE), onehot−softmax (CE)

        # ---- §24 (s32): A=JJᵀr (1st-order Δf) & B=(η/2N)·[(J·r)ᵀQ_k(J·r)]_k (2nd-order Δf) alignment ----
        # MIRRORS server.py:3216-3231. Gate: s32 and dataset!="owt" and Jc/rr available (server.py:3217).
        if self.s32 and self.dataset != "owt" and Jc is not None and rr is not None:
            from .sec24 import sec24_payload
            rec["g24"] = sec24_payload(self.model, Jc, rr, th, X, lr, N, outD)

        # ---- §25 (s33): ‖∇L‖ + d/dt(J·r) split (jdr/jrd) + II/III + 5-gradient cosine alignments ----
        # MIRRORS server.py:3238-3265. Gate: s33 and (N·outD)<=grid3dcap and dataset!="owt" and Jc/rr available.
        if (self.s33 and (N * outD) <= self.grid3dcap and self.dataset != "owt"
                and Jc is not None and rr is not None):
            from .sec25 import sec25_payload
            rec["g25"] = sec25_payload(self.model, self.loss, Jc, rr, th, X, Y, lr, N, outD,
                                       self._sec25_hist, t, rhist=self._sec25_rhist)

        # ---- §26 (s34): eigenvector-direction drift |cos(v_i(t),v_i(t−k))| of top-3 GN/NTK & top-3⊕bottom-3 M_r ----
        # MIRRORS server.py §26 block. Gate: s34 and (N·outD)<=grid3dcap and dataset!="owt" and Jc/rr available.
        if (self.s34 and (N * outD) <= self.grid3dcap and self.dataset != "owt"
                and Jc is not None and rr is not None):
            from .sec26 import sec26_eigvecs, sec26_drift
            ev26 = sec26_eigvecs(self.model, Jc, rr, th, X, N, outD)
            rec["g26"] = sec26_drift(ev26, self._sec26_hist, t)
            self._sec26_hist.append({"t": t, **{key: [v.detach().clone() for v in ev26[key]] for key in ev26}})
            if len(self._sec26_hist) > 101:
                self._sec26_hist.pop(0)

        # ---- §7a (NTK alignment, always-on) + §7 (heavy FH-tensor SVD projections) ----
        # §7a needs only the NTK eigvecs Vk and the FH-tensor eigvecs Vh; the §7 projections additionally
        # need the Gauss–Newton vecs UJ and the (expensive) FH eigenvectors via Lanczos.
        if (self.s7 or self.s13) and self.multi_ok:
            Kv, Vk = sym_eig_desc(Jc @ Jc.t())
            for k in range(n7):
                Vk[:, k] = sign_to(Vk[:, k], self._prev_ntk[k]); self._prev_ntk[k] = Vk[:, k]
            GH = torch.zeros(M, M, dtype=dt, device=dev)
            for pr in range(self.nprobe):
                Hz = jac_hvp(self.model, th, X, randn_vec(p, (0xC0FFEE + pr * 40503) & 0xFFFFFFFF, dev, dt))
                GH += Hz @ Hz.t()
            GH /= self.nprobe
            Hv, Vh = sym_eig_desc(GH)
            for k in range(n7):
                Vh[:, k] = sign_to(Vh[:, k], self._prev_vh[k]); self._prev_vh[k] = Vh[:, k]
            # §7a: residual onto top-n NTK eigvecs; top NTK eigvec onto FH right-singular vecs
            rec["ntkR"] = [float(rr @ Vk[:, k]) for k in range(n7)]
            rec["ntkH"] = [float(Vk[:, 0] @ Vh[:, k]) for k in range(n7)]
            # §7a products: Δsharp(t)·Δ⟨r,vₖ⟩ — synchronous and 1-step lagged — each with a running time-avg.
            # Δx(t)=x(t)-x(t-1).  gₖ=Δλ(t)·Δrₖ(t);  g'ₖ=Δλ(t)·Δrₖ(t-1) (residual change offset back one step).
            sh = rec.get("sharpness")
            dSh = (sh - self._prev_sharp7a) if (self._prev_sharp7a is not None and sh is not None) else None
            if dSh is not None and self._prev_ntkR7a is not None:
                self._cntGs7a += 1; gs = []; gsA = []
                for k in range(n7):
                    g = dSh * (rec["ntkR"][k] - self._prev_ntkR7a[k]); self._sumGsync7a[k] += g
                    gs.append(g); gsA.append(self._sumGsync7a[k] / self._cntGs7a)
                rec["ntkGs"] = gs; rec["ntkGsA"] = gsA
            if dSh is not None and self._prev_prev_ntkR7a is not None:
                self._cntGl7a += 1; gl = []; glA = []
                for k in range(n7):
                    g = dSh * (self._prev_ntkR7a[k] - self._prev_prev_ntkR7a[k]); self._sumGlag7a[k] += g
                    gl.append(g); glA.append(self._sumGlag7a[k] / self._cntGl7a)
                rec["ntkGl"] = gl; rec["ntkGlA"] = glA
            self._prev_prev_ntkR7a = self._prev_ntkR7a
            self._prev_ntkR7a = list(rec["ntkR"]); self._prev_sharp7a = sh

            if self.s7 and heavy_tick:      # heavy §7 projections (FH eigenvalues + J onto FH eigvecs) — throttled
                UJ = []
                for k in range(n7):
                    u = (Jc.t() @ Vk[:, k]) / math.sqrt(max(float(Kv[k]), 1e-30))
                    u = sign_to(u, self._prev_uj[k]); self._prev_uj[k] = u; UJ.append(u)
                nM = min(2, max(1, p // 2))
                # lanczos_extreme → (top_vecs, bot_vecs, top_vals, bot_vals)
                fe1tv, fe1bv, fe1tval, fe1bval = lanczos_extreme(
                    self._hS(th, X, Vh[:, 0].reshape(N, outD)), p, nM, self.mV, 0x9E37, dev, dt)
                gh1 = Vh[:, 1] if Vh.shape[1] > 1 else Vh[:, 0]
                fe2tv, fe2bv, fe2tval, fe2bval = lanczos_extreme(
                    self._hS(th, X, gh1.reshape(N, outD)), p, nM, self.mV, 0x7C19, dev, dt)

                def _l(a, i):
                    return a[i] if i < len(a) else a[0]
                m1 = [fe1tv[0], _l(fe1tv, 1), fe1bv[0], _l(fe1bv, 1)]    # FH eigvectors (for projections)
                m2 = [fe2tv[0], _l(fe2tv, 1), fe2bv[0], _l(fe2bv, 1)]
                for j in range(4):
                    m1[j] = sign_to(m1[j], self._prev_m1[j]); self._prev_m1[j] = m1[j]
                    m2[j] = sign_to(m2[j], self._prev_m2[j]); self._prev_m2[j] = m2[j]
                rec["fhEvT"] = [fe1tval[0], _l(fe1tval, 1), fe2tval[0], _l(fe2tval, 1)]   # FH eigenvalues
                rec["fhEvB"] = [fe1bval[0], _l(fe1bval, 1), fe2bval[0], _l(fe2bval, 1)]
                rec["jhe1"] = [float(UJ[k] @ m1[0]) for k in range(n7)]
                rec["jhe2"] = [float(UJ[k] @ m1[1]) for k in range(n7)]
                rec["jhe1b"] = [float(UJ[k] @ m1[2]) for k in range(n7)]
                rec["jhe2b"] = [float(UJ[k] @ m1[3]) for k in range(n7)]
                rec["jh2e1"] = [float(UJ[k] @ m2[0]) for k in range(n7)]
                rec["jh2e2"] = [float(UJ[k] @ m2[1]) for k in range(n7)]
                rec["jh2e1b"] = [float(UJ[k] @ m2[2]) for k in range(n7)]
                rec["jh2e2b"] = [float(UJ[k] @ m2[3]) for k in range(n7)]

        # ---- §8 vec(J) onto right singular vecs of the p×(p·dₙ) FH reshape (heaviest section → throttled) ----
        if self.s8 and self.multi_ok and heavy_tick:
            jfn = max(float(Jc.norm()), 1e-30)
            wv = torch.zeros(p, dtype=dt, device=dev)
            eps = 1e-3
            for a in range(M):
                wv += (grad_out(self.model, th + eps * Jc[a], X, a)
                       - grad_out(self.model, th - eps * Jc[a], X, a)) / (2 * eps)
            gt, _, gtv, _ = lanczos_extreme(lambda v: hvp_G2(self.model, th, X, v), p, n8, self.mV, 0xB2D4, dev, dt)
            g2J = []
            for k in range(n8):
                uk = sign_to(gt[k], self._prev_g2[k]); self._prev_g2[k] = uk
                g2J.append(float(wv @ uk) / math.sqrt(max(gtv[k], 1e-30)))
            rec["g2J"] = g2J
            rec["g2Jn"] = [x / jfn for x in g2J]

        # ---- §4b/§4c/§4d base: NTK top eigvec u₁ (deterministic sign), J·r ----
        u1s = Jrs = bEk_vecs = bEk_vals = None
        Jrns = Rn = 1.0
        if self.multi_ok and (self.s9 or self.s10 or self.s11 or self.s12 or self.s15 or self.s16 or self.s17 or self.s18):
            Kb, Vb = sym_eig_desc(Jc @ Jc.t())
            for k in range(min(n, Vb.shape[1])):   # NTK is M×M → only M eigvecs exist (guard M < neig, e.g. tiny §11 runs)
                Vb[:, k] = pin_sign(Vb[:, k])
            bEk_vecs, bEk_vals = Vb, Kb
            u1s = Vb[:, 0]
            Rn = max(float(rr.norm()), 1e-30)
            Jrs = Jc.t() @ rr
            Jrns = max(float(Jrs.norm()), 1e-30)

        # Q[u₁] eigvecs (shared by §4b and §4d), sign-tracked once
        qeT = qeB = None
        if self.multi_ok and (self.s9 or self.s11):
            qeT, qeB, _, _ = lanczos_extreme(self._hS(th, X, u1s.reshape(N, outD)), p, n, self.mV, 0xC0DE1, dev, dt)
            for k in range(n):
                qeT[k] = sign_to(qeT[k], self._prev_q9t[k]); self._prev_q9t[k] = qeT[k]
                qeB[k] = sign_to(qeB[k], self._prev_q9b[k]); self._prev_q9b[k] = qeB[k]

        # ---- §4b J·r onto eigvecs of Q[u₁]; alignment with GN top eigvec g₁ ----
        if self.s9 and self.multi_ok:
            g1 = Jc.t() @ u1s
            g1 = g1 / max(float(g1.norm()), 1e-30)
            q9t = [float(Jrs @ qeT[k]) for k in range(n)]
            q9b = [float(Jrs @ qeB[k]) for k in range(n)]
            rec["q9t"], rec["q9b"] = q9t, q9b
            rec["q9tN"] = [x / Jrns for x in q9t]; rec["q9bN"] = [x / Jrns for x in q9b]
            rec["q9tR"] = [x / Rn for x in q9t]; rec["q9bR"] = [x / Rn for x in q9b]
            rec["q9gt"] = [float(g1 @ qeT[k]) for k in range(n)]
            rec["q9gb"] = [float(g1 @ qeB[k]) for k in range(n)]

        # ---- §4c Q[u₁]·(J·r) onto GN eigvecs g_k = J v_k/‖·‖ ----
        if self.s10 and self.multi_ok:
            QJr = hvp_S(self.model, th, X, u1s.reshape(N, outD), Jrs)
            qn = max(float(QJr.norm()), 1e-30)
            q10 = []
            for k in range(n):
                g = Jc.t() @ bEk_vecs[:, k]
                g = g / max(float(g.norm()), 1e-30)
                q10.append(float(QJr @ g))
            rec["q10"] = q10
            rec["q10N"] = [x / qn for x in q10]

        # ---- §4d per-group J / J·r onto the shared H and Q[u₁] eigvecs ----
        if (self.s11 and self.multi_ok and feTV is not None and qeT is not None
                and pos_rows is not None):
            def _grp(rows, weighted):
                if rows.numel() == 0:
                    return torch.zeros(p, dtype=dt, device=dev)
                sub = Jc[rows]
                if weighted:
                    sub = rr[rows].unsqueeze(1) * sub
                return sub.sum(0)
            Jp, Jn_ = _grp(pos_rows, False), _grp(neg_rows, False)
            Jrp, Jrn = _grp(pos_rows, True), _grp(neg_rows, True)
            rec["d4Ht"] = [float(Jp @ feTV[k]) for k in range(n)] + [float(Jn_ @ feTV[k]) for k in range(n)]
            rec["d4Hb"] = [float(Jp @ feBV[k]) for k in range(n)] + [float(Jn_ @ feBV[k]) for k in range(n)]
            rec["d4Qt"] = [float(Jrp @ qeT[k]) for k in range(n)] + [float(Jrn @ qeT[k]) for k in range(n)]
            rec["d4Qb"] = [float(Jrp @ qeB[k]) for k in range(n)] + [float(Jrn @ qeB[k]) for k in range(n)]

        # ---- §9 theory vs empirical σ₁ over frozen-Q windows ----
        if self.s12 or self.s14 or self.s15 or self.s16:
            thP, thA, thPpsd, thP_d, thPpsd_d = self._theory_step(th, X, Y, t, J, out, rr, bEk_vals)
            rec["thP"] = [(x / N if x is not None else None) for x in thP] if thP else None
            rec["thA"] = [(x / N if x is not None else None) for x in thA] if thA else None
            rec["thPpsd"] = [(x / N if x is not None else None) for x in thPpsd] if thPpsd else None
            # §9d: same predictions with the quadratically-self-computed residual (vs §9's NTK σ₁ = thA, §9d-c's thAH)
            rec["thP_d"] = [(x / N if x is not None else None) for x in thP_d] if thP_d else None
            rec["thPpsd_d"] = [(x / N if x is not None else None) for x in thPpsd_d] if thPpsd_d else None
            if self.s14 or self.s16:   # §9c/§9d-c actual: full loss-Hessian sharpness λmax(∇²L) (reuse §1's if present)
                sh = rec.get("sharpness")
                if sh is None:
                    sh = float(lanczos_extreme_vals(self._hL(th, X, Y), self.p, 1, self.mV, SEED_L,
                                                    self.device, self.dtype)[0][0])
                rec["thAH"] = sh

        # ---- §10 CUBIC approximation: Eq-47 (single) / Eq-51 (multi) σ₁ predictions, exact J&Q propagation ----
        if self.s17:
            shH = rec.get("sharpness")     # panel-2 actual: full loss-Hessian λmax(∇²L) (reuse §1's if present)
            if shH is None:
                shH = float(lanczos_extreme_vals(self._hL(th, X, Y), self.p, 1, self.mV, SEED_L,
                                                 self.device, self.dtype)[0][0])
            self._cubic_step(th, X, Y, t, J, out, rr, bEk_vals, shH, rec)

        # ---- §11 3D grids over sample indices (i,j,k): T1=Jᵢᵀ Qⱼ Jₖ, T2=uⱼuₖ·T1, T3=rᵢuⱼuₖ·T1 ----
        # (u = top NTK eigenvector, r = residual). M³ points + M Hessian-vector products → small M only, computed
        # EVERY diagnostic tick (one grid snapshot per iteration, to resolve the EoS dynamics — mirrors the
        # widget). For each k: jac_hvp(Jₖ)={Qⱼ Jₖ}ⱼ, then T1[:,:,k] = Jc·{Qⱼ Jₖ}ⱼᵀ.
        if (self.s18 and cube_emit and self.multi_ok and Jc is not None and u1s is not None
                and rr is not None and N <= self.grid3dcap):
            # §11 uses ONE output per sample — the GROUND-TRUTH-LABEL output (argmax of the target). Single-output
            # is identity (Ms=N); multi-output (cifar/sorting) picks f_i[label_i] so (i,j,k) range over N samples.
            if outD > 1:
                lblsel = torch.arange(N, device=dev) * outD + Y.reshape(N, outD).argmax(dim=1)
                Jc11 = Jc[lblsel]; rr11 = rr[lblsel]
                Kx, Vx = sym_eig_desc(Jc11 @ Jc11.t()); u11 = pin_sign(Vx[:, 0])   # NTK eigvec over the N label-outputs
            else:
                lblsel = None; Jc11 = Jc; rr11 = rr; u11 = u1s
            u = u11; Ms = N
            T1 = torch.zeros(Ms, Ms, Ms, dtype=dt, device=dev)
            Gd = torch.zeros(Ms, Jc11.shape[1], dtype=dt, device=dev)   # gᵢ = Qᵢ Jᵢ (diagonal row of each HVP)
            for k in range(Ms):
                QJk = jac_hvp(self.model, th, X, Jc11[k])        # (M,p): {Q_a · J_sel[k]}_a over ALL model outputs
                QJr = QJk[lblsel] if lblsel is not None else QJk  # (Ms,p): ground-truth-label Hessian rows
                T1[:, :, k] = Jc11 @ QJr.t()                      # [i,j] = Jᵢ·(Qⱼ Jₖ)
                Gd[k] = QJr[k]                                    # Qₖ Jₖ (reused → no extra HVP)
            T2 = T1 * u.view(1, Ms, 1) * u.view(1, 1, Ms)        # uⱼ uₖ T1
            T3 = T2 * rr11.view(Ms, 1, 1)                         # rᵢ uⱼ uₖ T1
            # §11 SQUARE — S[i,j] = Jᵢᵀ Qᵢ Jⱼ · rⱼ = (Qᵢ Jᵢ)·Jⱼ · rⱼ (2D over (i,j); classes i=j / i≠j)
            Smat = (Gd @ Jc11.t()) * rr11.view(1, Ms)            # (Ms,Ms): [i,j] = gᵢ·Jⱼ · rⱼ
            jj_ = torch.arange(Ms, device=dev)
            cats = torch.where(jj_.view(Ms, 1) == jj_.view(1, Ms), 0, 1).reshape(-1)   # 0:i=j  1:i≠j

            def _evs(S):   # per (i=j / i≠j) class: [mean, std, mean|·|]
                f = S.reshape(-1).detach()
                out = []
                for c in range(2):
                    sel = f[cats == c]
                    out.append([float(sel.mean()), float(sel.std(unbiased=False)), float(sel.abs().mean())]
                               if int((cats == c).sum()) else [0.0, 0.0, 0.0])
                return out
            # Per-class stats (mean/std/mean|·|) + positive-share over the FULL grid for the 5 diagonal classes
            # of (i,j,k): 0:i=j=k 1:i=j≠k 2:i≠j=k 3:i=k≠j 4:i≠j≠k. Computed EVERY tick (tiny) so the evolution /
            # norm-share / abs-norm-share curve panels are dense like the widget.
            ai = torch.arange(Ms, device=dev)
            eij = ai.view(Ms, 1, 1) == ai.view(1, Ms, 1)
            ejk = ai.view(1, Ms, 1) == ai.view(1, 1, Ms)
            eik = ai.view(Ms, 1, 1) == ai.view(1, 1, Ms)
            catf = torch.where(eij & ejk, 0, torch.where(eij, 1, torch.where(ejk, 2,
                               torch.where(eik, 3, 4)))).reshape(-1)

            def _ev(T):   # per class: [mean, std, mean|·|] over the FULL grid
                f = T.reshape(-1).detach()
                out = []
                for c in range(5):
                    sel = f[catf == c]
                    if int((catf == c).sum()):
                        out.append([float(sel.mean()), float(sel.std(unbiased=False)), float(sel.abs().mean())])
                    else:
                        out.append([0.0, 0.0, 0.0])
                return out

            def _pospct(T):   # 100·(Σ positive)/(Σ positive + |Σ negative|) over the full N³ grid
                f = T.reshape(-1).detach()
                ps = float(f[f > 0].sum()); ns = float(f[f < 0].sum())   # ns ≤ 0 → |neg sum| = -ns
                den = ps - ns
                return ps / den * 100.0 if den > 1e-30 else 0.0

            g3d = {"M": Ms, "pp": [_pospct(T1), _pospct(T2), _pospct(T3)],
                   "ev": {"t1": _ev(T1), "t2": _ev(T2), "t3": _ev(T3)},
                   "sqpp": _pospct(Smat), "sqev": _evs(Smat)}

            # The full M³ grid (for the 3D scatter plots) is packed/stored EVERY diagnostic tick (mirrors the
            # widget's g3dstride=1) — dense per-iteration frames are essential to read the Edge-of-Stability
            # dynamics. This is cheap: the N HVPs that build T1/T2/T3 already ran above for the every-tick
            # curves, so only the reshape/top-k/CPU-copy is added here; per-frame size stays bounded by
            # grid3dcap + the top-|value| sparsification. Heavy-tailed (~90% near-zero) → keep only the
            # top-|value| points per grid once the cube exceeds the render budget, so M≈100 (M³≈10⁶) stays
            # renderable. idx = i·M² + j·M + k (each grid its own top set).
            sparse = (Ms * Ms * Ms) > G3D_MAXPTS

            def _pack(T):
                return _g3d_pack(T.reshape(-1))

            v1, i1 = _pack(T1); v2, i2 = _pack(T2); v3, i3 = _pack(T3)
            dd = torch.arange(Ms, device=dev)           # i=j=k diagonal values (stay coloured even when sparse)
            d1 = T1[dd, dd, dd]; d2 = T2[dd, dd, dd]; d3 = T3[dd, dd, dd]
            sqv, sqi = _g3d_pack(Smat.reshape(-1))     # the square is M² — same span-preserving subset
            sqsparse = sqi is not None
            g3d.update({"sparse": sparse, "t1": v1, "t2": v2, "t3": v3,
                        "d1": d1.detach().cpu().tolist(), "d2": d2.detach().cpu().tolist(), "d3": d3.detach().cpu().tolist(),
                        "sq": sqv, "sqsparse": sqsparse,                       # 4th column S[i,j], idx=i·M+j
                        "sqd": torch.diagonal(Smat).detach().cpu().tolist()})  # i=j diagonal (highlight; full M)
            if sqsparse:
                g3d["sqi"] = sqi
            if sparse:
                g3d.update({"i1": i1, "i2": i2, "i3": i3})
            rec["g3d"] = g3d

        # ---- §12 per-sample Hessian Q_i=∇²f_i eigenvector cross-similarity (ground-truth-label output) ----
        # MATRIX-FREE: extract each Q_i's top-K12 & bottom-K12 eigenpairs by Lanczos on v↦Q_i·v — no p×p
        # Hessian formed. For the MLP the N samples' Lanczos run IN ONE BATCH via a vmap'd HVP (one vectorised
        # forward/backward per step, ~4–8× faster on GPU); other archs fall back to the per-sample loop.
        if ((self.s19 or self.s20 or self.s21 or self.s22 or self.s26) and cube_emit and (self.multi_ok or self.s12single) and Jc is not None and rr is not None
                and N <= self.sec12ncap):
            lblsel12 = (torch.arange(N, device=dev) * outD + Y.reshape(N, outD).argmax(dim=1)
                        if outD > 1 else torch.arange(N, device=dev))
            K12 = max(1, min(SEC12_KFULL, p // 2))
            m12 = min(p, max(6 * K12, 64))                      # enough Lanczos steps to converge the K12-th eigenpair
            seeds12 = [(0x5EC12 + i * 7919) & 0x7FFFFFFF for i in range(N)]
            if getattr(self.model, "spec", None) is not None:   # MLP — batched vmap HVP
                import torch.func as _tf
                lbl12 = (Y.reshape(N, outD).argmax(dim=1) if outD > 1
                         else torch.zeros(N, dtype=torch.long, device=dev))
                OH = torch.zeros(N, outD, dtype=dt, device=dev)
                OH[torch.arange(N, device=dev), lbl12] = 1.0
                spec12 = self.model.spec
                def _f12(t, x, oh):
                    return (mlp_forward(t, spec12, x.unsqueeze(0)).reshape(-1) * oh).sum()
                def _hv12(t, x, oh, v):
                    return _tf.jvp(lambda tt: _tf.grad(_f12)(tt, x, oh), (t,), (v,))[1]
                hvp_batch = lambda V: _tf.vmap(_hv12, in_dims=(None, 0, 0, 0))(th, X, OH, V)
                TV, TW, BV, BW = batched_lanczos_extreme(hvp_batch, p, N, K12, m12, seeds12, dev, dt)
            else:
                TVl = []; TWl = []; BVl = []; BWl = []
                for i in range(N):
                    ci = torch.zeros(N, outD, dtype=dt, device=dev)
                    ci.view(-1)[int(lblsel12[i])] = 1.0
                    tv, bv, tval, bval = lanczos_extreme(
                        lambda v, ci=ci: hvp_S(self.model, th, X, ci, v), p, K12, m12, seeds12[i], dev, dt)
                    TVl.append(torch.stack(tv)); BVl.append(torch.stack(bv))
                    TWl.append(torch.tensor(tval, dtype=dt, device=dev))
                    BWl.append(torch.tensor(bval, dtype=dt, device=dev))
                TV, TW, BV, BW = torch.stack(TVl), torch.stack(TWl), torch.stack(BVl), torch.stack(BWl)
            r12 = rr[lblsel12]; Jg12 = Jc[lblsel12]
            gTV = gTW = gBV = gBW = None; wbases = None
            if self.s22:                   # panels 3/4: top-2/bottom-2 eigvecs of the AVERAGED Hessian (1/N)Σ_i Q_i
                cAvg = torch.zeros(N, outD, dtype=dt, device=dev)        # = ∇²((1/N)Σ_i f_i^GT) via hvp_S with cotangent 1/N
                cAvg.view(-1)[lblsel12] = 1.0 / N
                gtv, gbv, gtw, gbw = lanczos_extreme(lambda v: hvp_S(self.model, th, X, cAvg, v), p, 2, m12, 0x6105A1, dev, dt)
                gTV = torch.stack(gtv); gBV = torch.stack(gbv)
                gTW = torch.tensor(gtw, dtype=dt, device=dev); gBW = torch.tensor(gbw, dtype=dt, device=dev)
                # panels 5/6/7: top-2/bottom-2 eigvecs of the WEIGHTED-AVERAGE Hessian (Σ_i w_i Q_i)/(Σ_i w_i),
                #   w = |top NTK eigvec| (p5), |2nd NTK eigvec| (p6), unit-normalized random U[0,1] (p7). NTK = J Jᵀ (N×N).
                ntk = Jg12 @ Jg12.t()
                nevecs = safe_eigh(ntk).eigenvectors            # columns ascending
                v1 = nevecs[:, -1].abs()
                v2 = (nevecs[:, -2] if N >= 2 else nevecs[:, -1]).abs()
                rrng = mulberry32(0x7A9D07)                             # fixed seed ⇒ browser-identical random vector
                rvec = torch.tensor([rrng() for _ in range(N)], dtype=dt, device=dev)
                rvec = rvec / rvec.norm().clamp_min(1e-30)
                wbases = []
                for wi, wv in enumerate([v1, v2, rvec]):
                    cW = torch.zeros(N, outD, dtype=dt, device=dev)
                    cW.view(-1)[lblsel12] = wv / wv.sum().clamp_min(1e-30)   # weighted-average cotangent (Σ c_i = 1)
                    wtv, wbv, wtw, wbw = lanczos_extreme(lambda v, c=cW: hvp_S(self.model, th, X, c, v), p, 2, m12, (0x7B0001 + wi * 0x101), dev, dt)
                    wbases.append({"tv": torch.stack(wtv), "tw": torch.tensor(wtw, dtype=dt, device=dev),
                                   "bv": torch.stack(wbv), "bw": torch.tensor(wbw, dtype=dt, device=dev)})
            if self.s19 or self.s22 or self.s26:   # §12a (angles+diag-align) ⊕ §12b (proj) — both in one payload; §18 (s26) re-slices the proj arrays
                rec["g4d"] = sec12_payload(TV, TW, BV, BW, r12, Jg12, self.grid3dcap, gTV=gTV, gTW=gTW, gBV=gBV, gBW=gBW, wbases=wbases)
                if self.s26 and N == 2:        # §18 thin view-record (pure re-slice of the §12 proj arrays at sample 0 vs 1; N=2 only)
                    from .sec18 import sec18_payload
                    g18 = sec18_payload(rec["g4d"])
                    if g18 is not None:
                        rec["g18"] = g18
            if self.s21 and self.multi_ok and N <= self.grid3dcap:   # §13 (own toggle): exact ref J_iᵀQ_jJ_k·r_k (N HVPs) + G1/G2/G3 — multi-only (N³ cube)
                T1_13 = torch.zeros(N, N, N, dtype=dt, device=dev)
                QJ_list = []
                for kk in range(N):
                    QJk = jac_hvp(self.model, th, X, Jg12[kk])     # (M,p) = {Q_a·J_k}_a over all outputs
                    QJr = QJk[lblsel12] if outD > 1 else QJk       # (N,p) GT-label rows = {Q_i·J_kk}_i
                    QJ_list.append(QJr)
                    T1_13[:, :, kk] = Jg12 @ QJr.t()               # [i,j] = J_i·(Q_j J_k)
                ref13 = T1_13 * r12.view(1, 1, N)                  # exact reference · r_k
                cur13 = {"TV": TV, "TW": TW, "BV": BV, "BW": BW, "r": r12, "Jg": Jg12}
                if self._sec13_prev is not None:               # need t-1 for Q_i/Q_k & J',r → skip the first eig-tick
                    rec["g13"] = sec13_stats(cur13, self._sec13_prev, ref13, self.lr)
                    if self._sec13_qjprev is not None:         # panel 4: residual-reweighting asymmetry ‖A−B‖/‖A‖·100
                        QJp, rpv = self._sec13_qjprev["QJ"], self._sec13_qjprev["r"]   # QJp[j,i]=Q_{i,t}J_{j,t}, rpv=r_{·,t} at PREV tick t
                        A = torch.einsum('jip,i,j->p', QJp, r12, rpv)   # (Σ_i r_{i,t+1}Q_{i,t})(Σ_j J_{j,t} r_{j,t})
                        B = torch.einsum('jip,i,j->p', QJp, rpv, r12)   # (Σ_i r_{i,t}Q_{i,t})(Σ_j J_{j,t} r_{j,t+1})
                        nA = float(A.norm())
                        rec["g13"]["ndiff"] = (float((A - B).norm()) / nA * 100.0) if nA > 0 else 0.0
                self._sec13_prev = {k_: v_.detach() for k_, v_ in cur13.items()}
                self._sec13_qjprev = {"QJ": torch.stack(QJ_list).detach(), "r": r12.detach()}   # (N_j,N_i,p) → PREV for next tick
            if self.s20 and self.multi_ok:                     # §14 shares §12's per-sample eigenpairs — multi-only (N³ cubes)
                self._sec14_rhist.append(r12.detach())
                if len(self._sec14_rhist) > 5:
                    self._sec14_rhist.pop(0)
                cur14 = {"TV": TV, "TW": TW, "BV": BV, "BW": BW, "r": r12, "Jg": Jg12}
                if self._sec14_prev is not None:               # need t (prev) for u,σ,r_j,J_k; skip the first eig-tick (like §13)
                    rec["g14"] = sec14_payload(self._sec14_prev, cur14, self.lr, self.grid3dcap, self._sec14_rhist)
                self._sec14_prev = {k_: v_.detach() for k_, v_ in cur14.items()}

        # ---- §15: 2nd-difference decomposition of ‖J‖²_F & σ₁ (own toggle; multi-only, small-N, holds 3 ticks) ----
        if self.s23 and (N * outD) <= self.grid3dcap and Jc is not None and rr is not None:   # §15 — effective samples M=N·d_out (A/B are M×M)
            M15 = N * outD                                         # EFFECTIVE samples = N·d_out
            lblsel15 = torch.arange(M15, device=dev)              # ALL N·d_out outputs are the effective samples (not just the GT-label)
            Jg15 = Jc[lblsel15]; r15 = rr[lblsel15]                # (M,p) per-output grads ∇f_k ; (M,) residual r=y−f
            # ---- panel 5: per-effective-sample NORM evolution (mean ± std across the M outputs), at the current step t ----
            jn15 = Jg15.norm(dim=1)                                # ‖∇f_k‖   (per-output Jacobian-row norm)
            rn15 = r15.abs()                                       # |r_k|    (per-output residual)
            gn15 = rn15 * jn15                                     # ‖r_k∇f_k‖ (per-output loss-gradient term)
            acc15 = torch.zeros(M15, dtype=dt, device=dev)         # ‖Q_{t,k}‖_F via Hutchinson: E_v‖Q_k v‖²=tr(Q_k²)
            for pr in range(4):
                vv15 = randn_vec(p, (0x5EC15 + pr * 0x9E3779B1) & 0xFFFFFFFF, dev, dt)
                Qv15 = jac_hvp(self.model, th, X, vv15)[lblsel15]
                acc15 = acc15 + (Qv15 * Qv15).sum(dim=1)
            hn15 = torch.sqrt(acc15 / 4.0)
            _ms15 = lambda x: [float(x.mean()), float(x.std(unbiased=False))]
            rec["g15n"] = {"g": _ms15(gn15), "j": _ms15(jn15), "h": _ms15(hn15), "r": _ms15(rn15)}
            # ---- new Panel B: per-EFFECTIVE-sample top/bottom-eigvec projection ratio of ∇f_k on its OWN Q_k (k = one of M=N·d outputs) ----
            mB = min(p, 48); seedsB = [(0x5EC15B + k * 7919) & 0x7FFFFFFF for k in range(M15)]
            if getattr(self.model, "spec", None) is not None:    # MLP — batched vmap HVP (∇²f_k·v) at the current θ
                import torch.func as _tf
                XrepB = X.repeat_interleave(outD, dim=0)          # (M,indim): input for effective sample k = X[k//d_out]
                OHB = torch.eye(outD, dtype=dt, device=dev).repeat(N, 1)   # (M,d_out): row k one-hot at output c=k%d_out
                specB = self.model.spec
                def _fB(tt, x, oh): return (mlp_forward(tt, specB, x.unsqueeze(0)).reshape(-1) * oh).sum()
                def _hvB(tt, x, oh, v): return _tf.jvp(lambda q: _tf.grad(_fB)(q, x, oh), (tt,), (v,))[1]
                hvpB = lambda V: _tf.vmap(_hvB, in_dims=(None, 0, 0, 0))(th, XrepB, OHB, V)
                TVb, TWb, BVb, BWb = batched_lanczos_extreme(hvpB, p, M15, 1, mB, seedsB, dev, dt)
                AsumQJ = hvpB(Jg15).sum(0)                       # Σ_k Q_k J_k = ½·∇θ‖J‖²_F (batched per-sample HVP)
            else:                                                # other archs — per-effective-sample Lanczos loop (single-output cotangent)
                TVl = []; TWl = []; BVl = []; BWl = []
                AsumQJ = torch.zeros(p, dtype=dt, device=dev)
                for k in range(M15):
                    ck = torch.zeros(N * outD, dtype=dt, device=dev); ck[k] = 1.0; ck = ck.reshape(N, outD)
                    tv, bv, tw, bw = lanczos_extreme(lambda v, ck=ck: hvp_S(self.model, th, X, ck, v), p, 1, mB, seedsB[k], dev, dt)
                    TVl.append(torch.stack(tv)); BVl.append(torch.stack(bv))
                    TWl.append(torch.tensor(tw, dtype=dt, device=dev)); BWl.append(torch.tensor(bw, dtype=dt, device=dev))
                    AsumQJ = AsumQJ + hvp_S(self.model, th, X, ck, Jg15[k])      # Q_k J_k accumulated
                TVb, TWb, BVb, BWb = torch.stack(TVl), torch.stack(TWl), torch.stack(BVl), torch.stack(BWl)
            rec["g15b"] = sec15_panelB(TVb[:, 0], BVb[:, 0], TWb[:, 0], BWb[:, 0], Jg15)
            # §15 Panel 8 — correlation of A=∇θ‖∇θf‖²=∇θ tr(NTK)=2Σ_k Q_k J_k and B=∇θ‖∇θL‖²=2H∇L (both nablas wrt θ)
            Avec = 2.0 * AsumQJ
            gL = grad_loss(self.model, self.loss, th, X, Y)[0]            # ∇θL
            gLn = float(gL.norm())
            if gLn > 1e-30:
                Hu = (grad_loss(self.model, self.loss, th + FD_EPS * (gL / gLn), X, Y)[0]
                      - grad_loss(self.model, self.loss, th - FD_EPS * (gL / gLn), X, Y)[0]) / (2 * FD_EPS)
                Bvec = 2.0 * gLn * Hu                                     # B = 2·H·∇L (FD on the unit gradient, rescaled — matches server)
            else:
                Bvec = torch.zeros(p, dtype=dt, device=dev)
            An = float(Avec.norm()); Bn = float(Bvec.norm()); inner = float(Avec @ Bvec)
            jnorm2_k = (Jg15 * Jg15).sum(dim=1)                          # ‖J_k‖² per effective sample
            jn2 = float(jnorm2_k.sum())                                  # ‖J‖²_F = tr(NTK)
            Q1 = float(jnorm2_k.mean())                                  # ⟨‖∇f_k‖²⟩_k
            Q2 = float(((r15 * r15) * jnorm2_k).mean())                  # ⟨‖r_k∇f_k‖²⟩_k
            dq1 = dq2 = None
            if len(self._sec15_hist) >= 1 and self._sec15_hist[-1]["t"] == t - 1:
                Jp = self._sec15_hist[-1]["J"]; rp = self._sec15_hist[-1]["r"]; jn2p_k = (Jp * Jp).sum(dim=1)
                dq1 = Q1 - float(jn2p_k.mean()); dq2 = Q2 - float(((rp * rp) * jn2p_k).mean())
            D2corr = None
            if (len(self._sec15_hist) >= 2 and self._sec15_hist[-1]["t"] == t - 1
                    and self._sec15_hist[-2]["t"] == t - 2):
                D2corr = (jn2 - 2.0 * float((self._sec15_hist[-1]["J"] ** 2).sum())
                          + float((self._sec15_hist[-2]["J"] ** 2).sum()))
            rec["g15corr"] = {"jn2": jn2, "An": An, "gn2": gLn * gLn, "Bn": Bn, "inner": inner,
                              "cos": (inner / (An * Bn) if An > 1e-30 and Bn > 1e-30 else 0.0),
                              "D2": D2corr, "pred": 0.5 * self.lr * self.lr * inner,
                              "q1": Q1, "q2": Q2, "dq1": dq1, "dq2": dq2}
            if (len(self._sec15_hist) >= 2 and self._sec15_hist[-1]["t"] == t - 1
                    and self._sec15_hist[-2]["t"] == t - 2):       # 3 CONSECUTIVE GD steps (eigevery=1): the per-step 2nd-difference theory requires it
                tm1, tm2 = self._sec15_hist[-1], self._sec15_hist[-2]
                hvp1 = lambda v: jac_hvp(self.model, tm1["th"], X, v)[lblsel15]   # {Q_{t−1,k}·v}_k  (N,p)
                hvp2 = lambda v: jac_hvp(self.model, tm2["th"], X, v)[lblsel15]   # {Q_{t−2,k}·v}_k
                g15rec, Amat, Bmat, u1_15 = sec15_stats(hvp1, hvp2, Jg15, tm1["J"], tm2["J"], tm1["r"], tm2["r"], self.lr, N, diveps=self.divreg)
                rec["g15"] = g15rec
                hvp_at = lambda thx, v: jac_hvp(self.model, thx, X, v)[lblsel15]   # §15 panels 3/4 (Q̇≠0): VII/VIII + A'/B' (2N² extra HVPs)
                rec["g15p34"] = sec15_panel34(hvp_at, tm2["th"], Jg15, tm1["J"], tm2["J"], tm1["r"], tm2["r"], u1_15, Amat, Bmat, g15rec, self.lr, N, diveps=self.divreg)
            self._sec15_hist.append({"th": th.detach().clone(), "J": Jg15.detach(), "r": r15.detach(), "t": t})
            if len(self._sec15_hist) > 2:
                self._sec15_hist.pop(0)

        # ---- §19 (s27): one-step grad-norm change A_t + tr(NTK) — MSE+GD only, every tick (server.py:3843) ----
        if (self.s27 and self.optimizer == "gd" and self.loss.name == "mse"
                and (N * outD) <= self.grid3dcap and Jc is not None and rr is not None):
            from .sec19 import sec19_payload
            rec["g19"] = sec19_payload(self.model, Jc, rr, th, X, lr, N, outD)

        # ---- §20 (s28): M_r=Σr_kQ_k spectral histograms — any loss/optimizer, cube cadence (server.py:3846) ----
        if (self.s28 and cube_emit and (N * outD) <= self.grid3dcap
                and Jc is not None and rr is not None):
            from .sec20 import sec20_payload
            rec["g20"] = sec20_payload(self.model, Jc, rr, th, X, N, outD, self.s28k)

        # ---- §21 (s29): residual↔spectrum alignment (NTK + M_r + GN panels) — cube cadence (server.py:3849) ----
        if (self.s29 and cube_emit and (N * outD) <= self.grid3dcap
                and Jc is not None and rr is not None):
            from .sec21 import sec21_payload
            rec["g21"] = sec21_payload(self.model, Jc, rr, th, X, N, outD, self.s29k)

        # ---- §5 SLQ spectral densities (expensive → ~50 snapshots/run via slqStride) ----
        if self.s5 and slq_tick:
            ng = 80                        # density grid points — match the widget (server/browser use 80)
            rec["slq_H"] = slq_density(self._hF(th, X), p, self.nprobe, self.mSLQ, ng, SEED_SLQ_H, dev, dt)
            rec["slq_lossH"] = slq_density(self._hL(th, X, Y), p, self.nprobe, self.mSLQ, ng, SEED_SLQ_L, dev, dt)
            if self.gs:
                rec["slq_G"] = slq_density(self._hG(th, X), p, self.nprobe, self.mSLQ, ng, SEED_SLQ_G, dev, dt)
                rec["slq_S"] = slq_density(self._hS(th, X, cS), p, self.nprobe, self.mSLQ, ng, SEED_SLQ_S, dev, dt)
            # trace of each operator over training (Hutchinson). H is ÷N → per-sample scale (matches §1).
            ntr = max(self.nprobe, 4)
            rec["tr_H"] = hutch_trace(self._hF(th, X), p, ntr, 0x51, dev, dt) / self.N
            rec["tr_HL"] = hutch_trace(self._hL(th, X, Y), p, ntr, 0x54, dev, dt)
            if self.gs:
                rec["tr_G"] = hutch_trace(self._hG(th, X), p, ntr, 0x52, dev, dt)
                rec["tr_S"] = hutch_trace(self._hS(th, X, cS), p, ntr, 0x53, dev, dt)
        return rec

    def _theory_step(self, th, X, Y, t, J, out, rr, bEk_vals):
        """§9 Eq-13/21/22/23/29 predicted σ₁ over frozen-Q windows. MIRRORS server's s12 block.
        Returns (thP, thA, thPpsd): predicted / actual / (predicted + 2nd-order PSD term ‖ΔJᵀu₁‖²) σ₁,
        as 5-vectors (col-1 single-sample Eq-13; cols-2..5 multi = Eq-21/22/23/29)."""
        N, p = self.N, self.p
        lr, ee = self.lr, self.eigevery
        outD = self.outD
        dev, dt = self.device, self.dtype
        qapprox, qmode, tset = self.qapprox, self.qmode, self.tset
        etaN = lr / max(N, 1)
        reps_ = max(1, ee)
        multi = self.multi                    # true M>1 (single-sample theory needs M==1)
        if multi and not self.multi_ok:       # multi theory needs the M×p Jacobian — infeasible at this size
            return ([None] * 5,) * 5     # 5-tuple: caller unpacks thP, thA, thPpsd, thP_d, thPpsd_d

        if self._thT0 < 0 or (t - self._thT0) >= qapprox:     # window start: freeze θ₀, J₀, FH
            self._thT0 = t
            self._thTh0 = th.clone()
            self._thAcc1 = self._thAcc2 = 0.0
            self._thProd3 = self._thProd4 = self._thProd5 = 1.0
            self._thAccPSD = self._thAccPSD1 = 0.0
            if multi:
                self._thJ, _ = jac_cols(self.model, th, X)
                self._thFroz = fh_frozen(self.model, th, X, self.nprobe, min(self.n, self.M),
                                         2, self.mV, self.M, dev, dt)
                self._thBase = float(safe_eigvalsh(self._thJ @ self._thJ.t())[-1])
                if self.s15 or self.s16:   # §9d: freeze f₀,J₀; reset the quad-GD displacement & parallel accumulators
                    self._thJ0 = self._thJ.clone(); self._thF0 = (Y.reshape(-1) - rr).clone() if rr is not None else None
                    self._thDth = torch.zeros(p, dtype=dt, device=dev); self._thJ_d = self._thJ.clone()
                    self._thProd3_d = self._thProd4_d = self._thProd5_d = 1.0
                    self._thAcc2_d = self._thAccPSD_d = 0.0
                else:
                    self._thJ_d = None
            else:
                Jc0, o0 = grad_sum_f(self.model, th, X)
                self._thJp = Jc0.clone()
                self._thBase = float(Jc0 @ Jc0)
                if self.s15 or self.s16:   # §9d single-sample: freeze f₀,J₀; reset displacement & accumulators
                    self._thJp0 = Jc0.clone(); self._thF0s = float(o0.reshape(-1)[0])
                    self._thDth_s = torch.zeros(p, dtype=dt, device=dev); self._thJp_d = Jc0.clone()
                    self._thAcc1_d = self._thAccPSD1_d = 0.0
                else:
                    self._thJp_d = None

        thP = [None]*5      # display cols 1-5: Eq-13, Eq-21, Eq-22, Eq-23, Eq-29 (col-4 reads thProd5=Eq-23, col-5 reads thProd4=Eq-29)
        thA = [None]*5
        thPpsd = [None]*5   # §9b: prediction + accumulated 2nd-order PSD term ‖ΔJᵀu₁‖²
        if multi and self._thJ is not None and self._thFroz is not None and bEk_vals is not None:
            Jt, Fz = self._thJ, self._thFroz
            Kw, Vw = sym_eig_desc(Jt @ Jt.t())
            NV0 = min(self.n, self.M)
            for k in range(NV0):
                Vw[:, k] = pin_sign(Vw[:, k])
            sig1 = max(float(Kw[0]), 1e-30); sgT = math.sqrt(sig1); u1 = Vw[:, 0]
            v1 = Jt.t() @ u1; v1 = v1 / max(float(v1.norm()), 1e-30)
            sigAct = float(bEk_vals[0]); thA[1] = thA[2] = thA[3] = thA[4] = sigAct
            cap = 10 * max(sigAct, 1e-30)
            clmp = lambda x: min(max(x, 0.0), cap)

            def gnv(k):
                g = Jt.t() @ Vw[:, k]
                return g / max(float(g.norm()), 1e-30)

            def fhBil(vk):
                s = 0.0
                for i in range(len(Fz["gamma"])):
                    yu = float(Fz["y"][i] @ u1)
                    sj = sum(Fz["tau"][i][j] * float(v1 @ Fz["z"][i][j]) * float(vk @ Fz["z"][i][j])
                             for j in range(len(Fz["z"][i])))
                    s += Fz["gamma"][i] * sj * yu
                return s

            thP[1] = clmp(self._thBase + self._thAcc2)
            thP[2] = clmp(self._thBase * self._thProd3)
            thP[3] = clmp(self._thBase * self._thProd5)   # col-4 = Eq-23 (swapped)
            thP[4] = clmp(self._thBase * self._thProd4)   # col-5 = Eq-29 (swapped)
            # §9b: same predictions + the dropped 2nd-order PSD term Σ‖ΔJᵀu₁‖² (always ≥0 — a sharpening floor)
            thPpsd[1] = clmp(self._thBase + self._thAcc2 + self._thAccPSD)
            thPpsd[2] = clmp(self._thBase * self._thProd3 + self._thAccPSD)
            thPpsd[3] = clmp(self._thBase * self._thProd5 + self._thAccPSD)   # col-4 = Eq-23
            thPpsd[4] = clmp(self._thBase * self._thProd4 + self._thAccPSD)   # col-5 = Eq-29
            qv1 = hvp_S(self.model, self._thTh0, X, u1.reshape(N, outD), v1)  # Q[u₁]v₁ = (Σ_a u₁_a ∇²f_a)·v₁ via the weighted
            #   HVP (2 grad evals, not the full M×p jac_hvp) ⇒ the EXACT bilinear v₁ᵀQ[u₁]v_k = qv1·v_k (Eq-22 & Eq-23)
            pproj = float(rr @ u1)                         # col-3 (Eq-22): p_t = r·u₁ (the v₁-coeff of Jᵀr)
            self._thProd3 *= (1 + 2 * etaN * pproj * float(qv1 @ v1)) ** reps_   # σ_{t+1}=σ_t[1+2η (r·u₁) v₁ᵀQ[u₁]v₁]  (exact)
            S4 = 0.0; S5 = 0.0; NV = min(max(tset, 1), NV0)
            for vk in range(NV):
                sgv = math.sqrt(max(float(Kw[vk]), 1e-30)); rho = float(rr @ Vw[:, vk]); gk = gnv(vk)
                S4 += (sgv / sgT) * fhBil(gk) * rho        # col-5 (Eq-29): bilinear via the γτz eigendecomposition of Q (approx)
                S5 += (sgv / sgT) * float(qv1 @ gk) * rho  # col-4 (Eq-23): bilinear v₁ᵀQ[u₁]v_k computed directly (exact)
            self._thProd4 *= (1 + 2 * etaN * S4) ** reps_
            self._thProd5 *= (1 + 2 * etaN * S5) ** reps_   # col-4 (Eq-23): differs from Eq-29 only by the Q-decomposition approx
            for _ in range(reps_):                            # advance Eq-15: J += (η/N) Q[Jᵀr]
                QW = jac_hvp(self.model, self._thTh0, X, self._thJ.t() @ rr)
                qu = (u1.unsqueeze(1) * QW).sum(0)
                self._thAcc2 += 2 * etaN * sgT * float(v1 @ qu)
                self._thAccPSD += (etaN ** 2) * float(qu @ qu)    # §9b: ‖ΔJᵀu₁‖² = (η/N)²‖qu‖²
                self._thJ = self._thJ + etaN * QW
        elif (not multi) and self._thJp is not None:
            Jcur, ocur = grad_sum_f(self.model, th, X); sigAct = float(Jcur @ Jcur); thA[0] = sigAct
            cap0 = 10 * max(sigAct, 1e-30)
            thP[0] = min(max(self._thBase + self._thAcc1, 0.0), cap0)
            thPpsd[0] = min(max(self._thBase + self._thAcc1 + self._thAccPSD1, 0.0), cap0)   # §9b: + Σ‖ΔJ‖²
            rsc = float((Y.reshape(-1) - ocur)[0])
            for _ in range(reps_):
                QJ = hvp_F(self.model, self._thTh0, X, self._thJp)
                self._thAcc1 += 2 * lr * rsc * float(self._thJp @ QJ)
                self._thAccPSD1 += ((lr * rsc) ** 2) * float(QJ @ QJ)    # §9b: ‖ΔJ‖² = (η r)²‖QJ‖²
                self._thJp = self._thJp + lr * rsc * QJ

        # ---- §9d: SAME predictions as §9/§9b, but the residual is computed BY the frozen quadratic model ----
        # (self-contained: inside the window r is the quadratic-GD residual r_q = Y−f_quad(θ₀+Δθ), not the live r)
        thP_d = [None] * 5; thPpsd_d = [None] * 5
        if (multi and self._thJ_d is not None and self._thFroz is not None
                and bEk_vals is not None and self._thF0 is not None):   # _thF0 guard matches server.py
            Jd, Fz = self._thJ_d, self._thFroz
            dth = self._thDth
            HzD = jac_hvp(self.model, self._thTh0, X, dth)               # Q[Δθ] = {∇²f_a(θ₀)·Δθ}, (M,p)
            r_q = Y.reshape(-1) - (self._thF0 + self._thJ0 @ dth + 0.5 * (HzD @ dth))   # quad-model residual
            Kw, Vw = sym_eig_desc(Jd @ Jd.t())
            NV0 = min(self.n, self.M)
            for k in range(NV0):
                Vw[:, k] = pin_sign(Vw[:, k])
            sgT = math.sqrt(max(float(Kw[0]), 1e-30)); u1 = Vw[:, 0]
            v1 = Jd.t() @ u1; v1 = v1 / max(float(v1.norm()), 1e-30)
            cap = 10 * max(float(bEk_vals[0]), 1e-30)
            clmpd = lambda x: min(max(x, 0.0), cap)

            def gnv_d(k):
                g = Jd.t() @ Vw[:, k]
                return g / max(float(g.norm()), 1e-30)

            def fhBil_d(vk):
                s = 0.0
                for i in range(len(Fz["gamma"])):
                    yu = float(Fz["y"][i] @ u1)
                    sj = sum(Fz["tau"][i][j] * float(v1 @ Fz["z"][i][j]) * float(vk @ Fz["z"][i][j])
                             for j in range(len(Fz["z"][i])))
                    s += Fz["gamma"][i] * sj * yu
                return s

            thP_d[1] = clmpd(self._thBase + self._thAcc2_d)
            thP_d[2] = clmpd(self._thBase * self._thProd3_d)
            thP_d[3] = clmpd(self._thBase * self._thProd5_d)   # col-4 = Eq-23
            thP_d[4] = clmpd(self._thBase * self._thProd4_d)   # col-5 = Eq-29
            thPpsd_d[1] = clmpd(self._thBase + self._thAcc2_d + self._thAccPSD_d)
            thPpsd_d[2] = clmpd(self._thBase * self._thProd3_d + self._thAccPSD_d)
            thPpsd_d[3] = clmpd(self._thBase * self._thProd5_d + self._thAccPSD_d)
            thPpsd_d[4] = clmpd(self._thBase * self._thProd4_d + self._thAccPSD_d)
            qv1 = hvp_S(self.model, self._thTh0, X, u1.reshape(N, outD), v1)
            pproj = float(r_q @ u1)
            self._thProd3_d *= (1 + 2 * etaN * pproj * float(qv1 @ v1)) ** reps_
            S4 = 0.0; S5 = 0.0; NV = min(max(tset, 1), NV0)
            for vk in range(NV):
                sgv = math.sqrt(max(float(Kw[vk]), 1e-30)); rho = float(r_q @ Vw[:, vk]); gk = gnv_d(vk)
                S4 += (sgv / sgT) * fhBil_d(gk) * rho
                S5 += (sgv / sgT) * float(qv1 @ gk) * rho
            self._thProd4_d *= (1 + 2 * etaN * S4) ** reps_
            self._thProd5_d *= (1 + 2 * etaN * S5) ** reps_
            for _ in range(reps_):                                # advance the quad GD (Eq-15 with r_q) + Δθ
                g = self._thJ_d.t() @ r_q
                QW = jac_hvp(self.model, self._thTh0, X, g)
                qu = (u1.unsqueeze(1) * QW).sum(0)
                self._thAcc2_d += 2 * etaN * sgT * float(v1 @ qu)
                self._thAccPSD_d += (etaN ** 2) * float(qu @ qu)
                self._thJ_d = self._thJ_d + etaN * QW
                self._thDth = self._thDth + etaN * g
        elif (not multi) and self._thJp_d is not None:
            dth = self._thDth_s
            Qd = hvp_F(self.model, self._thTh0, X, dth)            # ∇²f(θ₀)·Δθ
            f_q = self._thF0s + float(self._thJp0 @ dth) + 0.5 * float(dth @ Qd)
            r_qs = float(Y.reshape(-1)[0] - f_q)
            cap0 = 10 * max(float(self._thJp_d @ self._thJp_d), 1e-30)
            thP_d[0] = min(max(self._thBase + self._thAcc1_d, 0.0), cap0)
            thPpsd_d[0] = min(max(self._thBase + self._thAcc1_d + self._thAccPSD1_d, 0.0), cap0)
            for _ in range(reps_):
                QJ = hvp_F(self.model, self._thTh0, X, self._thJp_d)
                self._thAcc1_d += 2 * lr * r_qs * float(self._thJp_d @ QJ)
                self._thAccPSD1_d += ((lr * r_qs) ** 2) * float(QJ @ QJ)
                self._thJp_d = self._thJp_d + lr * r_qs * QJ
                self._thDth_s = self._thDth_s + lr * r_qs * self._thJp_d
        return thP, thA, thPpsd, thP_d, thPpsd_d

    # ----------------------------------------------------------------------------------------------------
    def _cubic_step(self, th, X, Y, t, J, out, rr, bEk_vals, shH, rec):
        """§10 CUBIC approximation (paper §6). At each `cubicapprox` window start freeze θ₀, J₀ and the
        3rd-derivative tensor T (accessed matrix-free as central differences of the HVPs); within the window
        propagate BOTH J and Q exactly — Q via the residual·Jacobian history, so a window costs O(window²)
        HVPs — and accumulate the per-step NTK-σ₁ change:
        with the TRUE-Taylor ΔJ = Q_t·Δθ + ½T[Δθ,Δθ], Δθ=η̃·Jᵀr (η̃=η single, η/N multi). Then
          single sample  ΔNTK = 2JᵀΔJ + ‖ΔJ‖²(PSD)
          multi  sample  Δσ₁  = 2√σ₁·v₁ᵀ(ΔJᵀu₁) + ‖ΔJᵀu₁‖²(PSD)
        Also the "without η²" curve: the SAME cubic trajectory & PSD, dropping only the explicit cubic
        interaction scalar η̃²√σ₁·v₁ᵀT[u₁](g,g) from Δσ₁; plus the window drift ‖ΔQ‖/‖Q₀‖, ‖ΔJ‖/‖J₀‖.
        Writes per-step keys into `rec` (÷N like §9):  c47/c47p, c51/c51p, c51n/c51np, cActN (NTK σ₁/N),
        cActH (loss-Hessian λmax), cdQ, cdJ.  Residual taken from the live run; everything else from the
        frozen quantities.  MIRRORS server's s17 block."""
        N, p, M = self.N, self.p, self.M
        lr, ee, outD = self.lr, self.eigevery, self.outD
        dev, dt = self.device, self.dtype
        etaN = lr / max(N, 1)
        reps_ = max(1, ee)
        eps = 1e-2                                  # unit-direction step for the 3rd-derivative finite differences
        multi = self.multi
        if multi and not self.multi_ok:
            return

        th0 = self._cTh0
        hvpF = lambda thx, v: hvp_F(self.model, thx, X, v)
        jhvp = lambda thx, v: jac_hvp(self.model, thx, X, v)

        def T2s(a, b):                              # single: T[a,b] = ∇³f(θ₀)[a,b] (p,) via central diff of Q·b
            an = float(a.norm())
            if an < 1e-30:
                return torch.zeros(p, dtype=dt, device=dev)
            ah = a / an
            return an * (hvpF(th0 + eps * ah, b) - hvpF(th0 - eps * ah, b)) / (2 * eps)

        def T2m(a, b):                              # multi: T[a,b] = {∇³f_c(θ₀)[a,b]}_c (M,p) via central diff of Q[b]
            an = float(a.norm())
            if an < 1e-30:
                return torch.zeros(M, p, dtype=dt, device=dev)
            ah = a / an
            return an * (jhvp(th0 + eps * ah, b) - jhvp(th0 - eps * ah, b)) / (2 * eps)

        # -------- window start: freeze θ₀, J₀, Hutchinson probes; reset propagated J / histories / accumulators
        if self._cT0 < 0 or (t - self._cT0) >= self.cubicapprox:
            self._cT0 = t
            self._cTh0 = th.clone(); th0 = self._cTh0
            self._cZ = [randn_vec(p, (0xCB1C + i * 0x9E3779B1) & 0xFFFFFFFF, dev, dt) for i in range(self._cn)]
            if multi:
                J0, _ = jac_cols(self.model, th0, X)
                self._cJ0 = J0
                self._cJ = J0.clone()
                self._cHistG = []
                self._cBase = float(bEk_vals[0]) if bEk_vals is not None else float(
                    safe_eigvalsh(J0 @ J0.t())[-1])
                self._cQ0z = [jhvp(th0, z) for z in self._cZ]              # Q₀·z_i (M,p)
                self._cdQz = [torch.zeros(M, p, dtype=dt, device=dev) for _ in self._cZ]
                self._cAcc51 = self._cAccPSD51 = self._cAcc51n = self._cAccPSD51n = 0.0
            else:
                J0 = J                                                    # grad_sum_f at θ₀ (this very step)
                self._cJ0 = J0.clone(); self._cJp = J0.clone()
                self._cHistR = []; self._cHistJ = []
                self._cBase = float(J0 @ J0)
                self._cQ0z = [hvpF(th0, z) for z in self._cZ]             # Q₀·z_i (p,)
                self._cdQz = [torch.zeros(p, dtype=dt, device=dev) for _ in self._cZ]
                self._cAcc47 = self._cAccPSD47 = 0.0

        cActH = shH                                                       # panel-2 actual: loss-Hessian λmax
        # Hutchinson ‖ΔQ‖/‖Q₀‖ (%) — common to both branches
        q0n = math.sqrt(max(sum(float(z @ z if z.dim() == 1 else (z * z).sum()) for z in self._cQ0z), 1e-30))
        dqn = math.sqrt(max(sum(float((z * z).sum()) for z in self._cdQz), 0.0))
        cdQ = 100.0 * dqn / q0n

        if multi and self._cJ is not None:
            actN = float(bEk_vals[0]); cap = 10.0 * max(actN, 1e-30)
            clmp = lambda x: min(max(x, 0.0), cap)
            rec["c51"] = clmp(self._cBase + self._cAcc51) / N
            rec["c51p"] = clmp(self._cBase + self._cAcc51 + self._cAccPSD51) / N
            rec["c51n"] = clmp(self._cBase + self._cAcc51n) / N
            rec["c51np"] = clmp(self._cBase + self._cAcc51n + self._cAccPSD51n) / N
            rec["cActN"] = actN / N
            rec["cActH"] = cActH
            rec["cdJ"] = 100.0 * float((self._cJ - self._cJ0).norm()) / max(float(self._cJ0.norm()), 1e-30)
            rec["cdQ"] = cdQ
            for _ in range(reps_):
                # ---- cubic recursion: Δθ=(η/N)Jᵀr; J & Q both propagated, only T frozen at θ₀ ----
                Jc = self._cJ
                Kc, Vc = sym_eig_desc(Jc @ Jc.t())
                u1 = pin_sign(Vc[:, 0]); sg = math.sqrt(max(float(Kc[0]), 1e-30))
                v1 = Jc.t() @ u1; v1 = v1 / max(float(v1.norm()), 1e-30)
                g = Jc.t() @ rr                                            # (p,) GD direction Jᵀr
                Qg = jhvp(th0, g)                                          # Q₀[g]
                for gk in self._cHistG:                                    # + (η/N) Σ_k T[g_k, g]  (Q propagated, Eq-50)
                    Qg = Qg + etaN * T2m(gk, g)
                Tgg = T2m(g, g)                                            # T[g,g]  (M,p)
                Qgu = (u1.unsqueeze(1) * Qg).sum(0)                        # Q_t[g]ᵀu₁  (p,)
                Tggu = (u1.unsqueeze(1) * Tgg).sum(0)                      # T[u₁](g,g)  (p,)
                dJ = etaN * Qg + 0.5 * (etaN ** 2) * Tgg                   # ΔJ = (η/N)Q_t[g] + ½(η/N)²T[g,g]  (true Taylor ½)
                dJu = dJ.t() @ u1                                          # ΔJᵀu₁  (p,)
                # Eq-51  Δσ₁ = 2√σ₁ v₁ᵀ(ΔJᵀu₁) + ‖ΔJᵀu₁‖² :  cubic = full ΔJ (the ½(η/N)²T term is the η² piece)
                self._cAcc51 += 2 * etaN * sg * float(v1 @ Qgu) + (etaN ** 2) * sg * float(v1 @ Tggu)
                self._cAccPSD51 += float(dJu @ dJu)                        # PSD ‖ΔJᵀu₁‖²
                # "without η²": SAME cubic trajectory & PSD, but DROP only the explicit cubic interaction scalar
                # (η/N)²√σ₁·v₁ᵀT[u₁](g,g) from Δσ₁ — so c51 − c51n isolates exactly that term.
                self._cAcc51n += 2 * etaN * sg * float(v1 @ Qgu)
                self._cAccPSD51n += float(dJu @ dJu)
                for i, z in enumerate(self._cZ):                          # ΔQ drift (Eq-50): ΔQ·z += (η/N) T[g,z]
                    self._cdQz[i] = self._cdQz[i] + etaN * T2m(g, z)
                self._cHistG.append(g)
                self._cJ = Jc + dJ
        elif (not multi) and self._cJp is not None:
            actN = float(J @ J); cap = 10.0 * max(actN, 1e-30)            # NTK σ₁ = ‖J‖² (single sample)
            clmp = lambda x: min(max(x, 0.0), cap)
            rec["c47"] = clmp(self._cBase + self._cAcc47)
            rec["c47p"] = clmp(self._cBase + self._cAcc47 + self._cAccPSD47)
            rec["cActN"] = actN
            rec["cActH"] = cActH
            rec["cdJ"] = 100.0 * float((self._cJp - self._cJ0).norm()) / max(float(self._cJ0.norm()), 1e-30)
            rec["cdQ"] = cdQ
            rsc = float((Y.reshape(-1) - out.reshape(-1))[0])             # live scalar residual y − f
            for _ in range(reps_):
                Jc = self._cJp
                QJ = hvpF(th0, Jc)                                        # Q₀·J
                for rk, Jk in zip(self._cHistR, self._cHistJ):            # + η Σ_k r_k T[J_k, J]  (Q propagated, Eq-45)
                    QJ = QJ + lr * rk * T2s(Jk, Jc)
                TJJ = T2s(Jc, Jc)                                         # T[J,J]
                dJ = lr * rsc * QJ + 0.5 * (lr ** 2) * (rsc ** 2) * TJJ   # ΔJ = ηr·Q_t·J + ½η²r²·T[J,J]  (true Taylor ½)
                # ΔNTK = 2JᵀΔJ + ‖ΔJ‖²
                self._cAcc47 += 2 * lr * rsc * float(Jc @ QJ) + (lr ** 2) * (rsc ** 2) * float(Jc @ TJJ)
                self._cAccPSD47 += float(dJ @ dJ)                         # PSD ‖ΔJ‖²
                for i, z in enumerate(self._cZ):                          # ΔQ drift (Eq-45): ΔQ·z += η r T[J,z]
                    self._cdQz[i] = self._cdQz[i] + lr * rsc * T2s(Jc, z)
                self._cHistR.append(rsc); self._cHistJ.append(Jc.clone())
                self._cJp = Jc + dJ
