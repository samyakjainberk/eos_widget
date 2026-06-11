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
                     jac_cols, jac_hvp, grad_out)
from .linalg import (lanczos_extreme, lanczos_extreme_vals, slq_density, sym_eig_desc,
                     sign_to, pin_sign, pos_subspace, neg_subspace, principal_angles, randn_vec)

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
        if loss.name == "ce":
            # CE uses the generic residual r = −∂L/∂z = softmax(z)−onehot (the loss-output cotangent). §4
            # (J→H) and §6 (rotation) are model-only; §7/§7a/§8/§4b/§4c use the function NTK (Jᵤ·Jᵤᵀ) plus
            # this residual, so they're all valid for CE — and gated by the same `multi_ok` size budget
            # (feasible for e.g. CIFAR-CE, M=N·classes; skipped at LM vocab sizes). Off for CE: §4d (its
            # sign-groups need a scalar residual) and §9 (the Eq-13/21/22/23/29 σ₁ recursion is squared-loss).
            self.s11 = self.s12 = False
        p = model.p
        half = max(1, p // 2)
        self.p = p
        self.N = cfg.nsamp
        self.outD = model.oc
        self.M = self.N * self.outD
        self.multi = self.M > 1
        # multi-sample sections form the M×p Jacobian — gate by a size budget
        self.multi_ok = self.multi and self.M <= max_M and self.M * p <= max_Mp
        self.n = min(max(1, cfg.neig), half)
        self.kth = min(max(1, cfg.kth), half)
        self.nSub = min(max(self.n, self.kth, (10 if self.s6 else 1)), half)
        self.n7 = min(self.n, max(self.M, 1))
        self.n8 = min(self.n, max(1, p // 2))
        self.nResid = min(self.M, 12)
        self.efrac = min(1.0, max(0.5, cfg.energyp / 100.0))
        self.nprobe = max(1, cfg.slqprobes)
        self.lr = cfg.lr
        self.thr = 2.0 / cfg.lr                      # edge-of-stability threshold 2/η
        self.eigevery = max(1, cfg.eigevery)
        self.qapprox = max(1, cfg.qapprox)
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
        th = theta
        N, outD, M, p = self.N, self.outD, self.M, self.p
        n, nSub, n7, n8 = self.n, self.nSub, self.n7, self.n8
        dev, dt = self.device, self.dtype
        lr = self.lr

        J, out = grad_sum_f(self.model, th, X)              # J = ∇Σf (p,)
        loss_val = float(self.loss.value(out, Y, N))
        cS = self.loss.resid_cotangent(out, Y, N)           # ∂L/∂out (feeds the S term + SLQ-S)
        rec = {"t": t, "loss": loss_val, "thr": self.thr}
        if self.loss.name != "ce" and self.dataset != "owt":  # CE, and the OWT LM under ANY loss, have no
            r = (Y - out).reshape(-1)                          #   scalar residual y−f (Y are token ids, shape ≠ out)
            rec["resid_mean"] = float(r.mean())
            rec["resid_rms"] = float(r.pow(2).mean().sqrt())
            rec["resid_head"] = r[:self.nResid].detach().cpu().tolist()

        need_vecs = self.s4 or self.s6
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

        # ---- §4 J onto top-/bottom-n eigvecs of H (raw + Frobenius-normalized) ----
        if self.s4 and feTV is not None:
            jn = max(float(J.norm()), 1e-30)
            jt = [float(J @ feTV[i]) for i in range(n)]
            jb = [float(J @ feBV[i]) for i in range(n)]
            rec["jt"], rec["jb"] = jt, jb
            rec["jtN"] = [x / jn for x in jt]
            rec["jbN"] = [x / jn for x in jb]

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
                                         or self.s12 or self.s13)
        if want_multi:
            Jc, out_flat = jac_cols(self.model, th, X)
            rr = (-N * cS).reshape(-1)        # generic residual −N·∂L/∂out: Y−f (MSE), onehot−softmax (CE)

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

            if self.s7:                     # heavy §7 projections (FH eigenvalues + J onto FH eigvecs)
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

        # ---- §8 vec(J) onto right singular vecs of the p×(p·dₙ) FH reshape ----
        if self.s8 and self.multi_ok:
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
        if self.multi_ok and (self.s9 or self.s10 or self.s11 or self.s12):
            Kb, Vb = sym_eig_desc(Jc @ Jc.t())
            for k in range(n):
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
        if self.s12 or self.s14:
            thP, thA, thPpsd, thP_d, thPpsd_d = self._theory_step(th, X, Y, t, J, out, rr, bEk_vals)
            rec["thP"] = [(x / N if x is not None else None) for x in thP] if thP else None
            rec["thA"] = [(x / N if x is not None else None) for x in thA] if thA else None
            rec["thPpsd"] = [(x / N if x is not None else None) for x in thPpsd] if thPpsd else None
            # §9d: same predictions with the quadratically-self-computed residual (vs §9's NTK σ₁ = thA, §9d-c's thAH)
            rec["thP_d"] = [(x / N if x is not None else None) for x in thP_d] if thP_d else None
            rec["thPpsd_d"] = [(x / N if x is not None else None) for x in thPpsd_d] if thPpsd_d else None
            if self.s14:   # §9c actual: full loss-Hessian sharpness λmax(∇²L) (per-sample; reuse §1's if present)
                sh = rec.get("sharpness")
                if sh is None:
                    sh = float(lanczos_extreme_vals(self._hL(th, X, Y), self.p, 1, self.mV, SEED_L,
                                                    self.device, self.dtype)[0][0])
                rec["thAH"] = sh

        # ---- §5 SLQ spectral densities (expensive) ----
        if self.s5:
            ng = 160
            rec["slq_H"] = slq_density(self._hF(th, X), p, self.nprobe, self.mSLQ, ng, SEED_SLQ_H, dev, dt)
            rec["slq_lossH"] = slq_density(self._hL(th, X, Y), p, self.nprobe, self.mSLQ, ng, SEED_SLQ_L, dev, dt)
            if self.gs:
                rec["slq_G"] = slq_density(self._hG(th, X), p, self.nprobe, self.mSLQ, ng, SEED_SLQ_G, dev, dt)
                rec["slq_S"] = slq_density(self._hS(th, X, cS), p, self.nprobe, self.mSLQ, ng, SEED_SLQ_S, dev, dt)
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
            return [None] * 5, [None] * 5, [None] * 5

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
                self._thBase = float(torch.linalg.eigvalsh(self._thJ @ self._thJ.t())[-1])
                # §9d: freeze f₀,J₀; reset the quad-GD displacement & parallel accumulators
                self._thJ0 = self._thJ.clone(); self._thF0 = (Y.reshape(-1) - rr).clone()
                self._thDth = torch.zeros(p, dtype=dt, device=dev); self._thJ_d = self._thJ.clone()
                self._thProd3_d = self._thProd4_d = self._thProd5_d = 1.0
                self._thAcc2_d = self._thAccPSD_d = 0.0
            else:
                Jc0, o0 = grad_sum_f(self.model, th, X)
                self._thJp = Jc0.clone()
                self._thBase = float(Jc0 @ Jc0)
                # §9d single-sample: freeze f₀,J₀; reset displacement & accumulators
                self._thJp0 = Jc0.clone(); self._thF0s = float(o0.reshape(-1)[0])
                self._thDth_s = torch.zeros(p, dtype=dt, device=dev); self._thJp_d = Jc0.clone()
                self._thAcc1_d = self._thAccPSD1_d = 0.0

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
        if multi and self._thJ_d is not None and self._thFroz is not None and bEk_vals is not None:
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
