"""§24 (s32) — A = J Jᵀr & B = (η/2N)·[(J·r)ᵀ Q_k (J·r)]_k alignment with the residual and the
top-4 NTK eigenvectors.  MIRRORS server.run_stream's `g24` block (server.py:3216-3231) EXACTLY.

Geometry.  At the current θ let J (M,p) be the per-output Jacobian (row k = ∇f_k, M=N·outD effective
samples) and r (M,) the generic residual (Y−f for MSE).  Then:

  * J·r = Jᵀ r = Σ_k r_k ∇f_k  ∈ ℝ^p   — the (un-normalised) loss gradient direction,
  * A = J Jᵀ r = NTK · r        ∈ ℝ^M   — the FIRST-order change in the network outputs f under a GD
        step (the linearised Δf), i.e. the NTK acting on the residual,
  * B = (η/2N)·[ (J·r)ᵀ Q_k (J·r) ]_k  ∈ ℝ^M   — the SECOND-order (curvature) correction to Δf, where
        Q_k = ∇²f_k is the per-sample function Hessian; the bracket is the per-output quadratic form of
        the gradient direction against each sample's curvature, scaled by η/2N.

Reported (all matched to server `g24`):
  * nA = ‖A‖, nB = ‖B‖, rn = ‖r‖     (each floored at 1e-30),
  * cAr = |⟨A,r⟩|/(‖A‖‖r‖), cBr = |⟨B,r⟩|/(‖B‖‖r‖)     — |cosine| of A,B with the residual,
  * cAu[i] = |⟨A,u_i⟩|/‖A‖, cBu[i] = |⟨B,u_i⟩|/‖B‖ for the top-4 NTK eigvecs u_i (cols of the eig of
        the M×M NTK Gram J Jᵀ, descending), padded with None to length 4,
  * sig[i] = λ_i(NTK)/N for the top-4 NTK eigvalues (÷N, as in §21), padded with None to length 4.

PARITY.  A, B, and the NTK eigenpairs are built from `jac_cols`/`jac_hvp`/`sym_eig_desc`, which mirror
server's definitions (jac_hvp is central-difference FD on BOTH sides with the same step EPS=1e-3), so
every field matches server to FD round-off in fp64 (~1e-10..1e-13).  This is a per-step diagnostic
PAYLOAD (returns the record dict, no yielding/side effects) invoked from diagnostics.compute(), matching
how sec12_payload / sec15_stats are already called — see the AUTHORITATIVE PORT SPEC §2(A).

Gate (server.py:3217): s32 and dataset != "owt" and Jc is not None and rr is not None.  (dataset!="owt"
and the Jc/rr availability are enforced by the caller; this function assumes Jc/rr are already built.)
"""
import torch

from .models import jac_hvp
from .linalg import sym_eig_desc


def sec24_payload(model, Jc, rr, th, X, lr, N, outD):
    """Compute the §24 `g24` record. MIRRORS server.py:3218-3231 line-for-line.

    Args:
      model : the network (eos_lab Model); passed explicitly (eos_lab is not module-global like server _TL).
      Jc    : per-output Jacobian J (M',p) from jac_cols(model, th, X)[0]  (M' may exceed M for GPT-LM; sliced to M).
      rr    : generic residual (M',) = Y−f (MSE).  Sliced to M.
      th    : current flat parameters θ (p,).
      X     : inputs.
      lr    : learning rate η.
      N     : effective minibatch sample count (loss-normalisation count).
      outD  : output dimension; M = N·outD.

    Returns:
      dict g24 = {nA, nB, rn, cAr, cBr, cAu[4], cBu[4], sig[4]}  (cAu/cBu/sig padded with None to length 4).
    """
    M = N * outD
    Jg24 = Jc[:M]                                              # M=N·outD effective-sample rows (slice guards GPT-LM, as §20/§21)
    r24 = rr[:M]
    Jr24 = Jg24.t() @ r24                                      # J·r = Σ_k r_k ∇f_k  (p,)
    A24 = Jg24 @ Jr24                                          # A = J Jᵀ r = NTK·r  (M,)
    B24 = (lr / (2.0 * N)) * (jac_hvp(model, th, X, Jr24)[:M] @ Jr24)   # B = (η/2N)·[(J·r)ᵀ Q_k (J·r)]_k  (M,)
    sg24v, U24 = sym_eig_desc(Jg24 @ Jg24.t())                # NTK Gram eigvals (desc) + eigvecs u_i (cols)
    nA24 = max(float(A24.norm()), 1e-30)
    nB24 = max(float(B24.norm()), 1e-30)
    rn24 = max(float(r24.norm()), 1e-30)
    nu24 = min(4, M)
    return {"nA": nA24, "nB": nB24, "rn": rn24,
            "cAr": abs(float(A24 @ r24)) / (nA24 * rn24),
            "cBr": abs(float(B24 @ r24)) / (nB24 * rn24),
            "cAu": [abs(float(A24 @ U24[:, i])) / nA24 for i in range(nu24)] + [None] * (4 - nu24),
            "cBu": [abs(float(B24 @ U24[:, i])) / nB24 for i in range(nu24)] + [None] * (4 - nu24),
            "sig": [float(sg24v[i]) / N for i in range(nu24)] + [None] * (4 - nu24)}   # NTK eigval ÷N (as §21)
