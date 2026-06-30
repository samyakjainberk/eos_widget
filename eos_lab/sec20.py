"""§20 (s28) — spectral histograms of the RESIDUAL-WEIGHTED function Hessian M_r = Σ_k r_k Q_k.

Standalone payload module, MIRRORS server._sec20_payload (server.py:1709-1755) and index.html s20Compute.

From the top-K ⊕ bottom-K eigenpairs (λ_i, v_i) of M_r = Σ_k r_k Q_k  (Q_k = ∇²f_k, the per-output
function Hessian) — obtained matrix-free via Lanczos on hvpS(·, r) started from the cross-backend
mulberry32+gauss vector _randvec16(p, SEC20_SEED) — we report, with scN = 1/N:
    lam      : {λ_i · scN}                              (Plot 1 histogram)
    q1/q2/q3/q4 : {λ_i · scN · |⟨v_i, u_k⟩|}  for k=1..4 (Plots 2/3/4/5 histograms)
where u_k = the top-k RIGHT-singular vector of J (= normalize(Jᵀ g_k), g_k the k-th eigenvector of the
NTK = J Jᵀ) — the "topmost eigenvectors of the Jacobian", in the same θ-space as v_i. |⟨·⟩| is gauge-free
(eigvec signs arbitrary). u_k is GATED on the NTK eigenvalue Kc[k] > 1e-9·Kc[0] (a near-zero σ_k² makes
Jᵀg_k round-off noise — backend-divergent if normalized), else the zero vector.

Lanczos uses the SAME q0 / T-eig / top-K⊕bottom-K-DISTINCT extraction as server / index.html ⇒ identical
M_r eigenpairs across backends (M_r's spectrum is clustered, so at finite Lanczos m the eigenpair estimate
is start-vector-dependent — the shared mulberry32 start makes it bit-consistent). Any loss / optimizer
(M_r is well-defined from the generic residual). Evolves over training (browser slider).

This module is SELF-CONTAINED: it carries its own q0-Lanczos core (a copy of linalg._lanczos_core extended
with the q0 start) so it parity-verifies WITHOUT any edit to linalg.py. At integration time the diagnostics
caller invokes sec20_payload(model, Jc, rr, th, X, N, outD, K) per step (cube cadence); the surrogate
drivers (§22/§23) pass mr_hvp=<surrogate M_r operator>.
"""
import torch

from .models import hvp_S
from .linalg import safe_eigh, sym_eig_desc
from .sec16 import _randvec16


SEC20_SEED = 0x205EE0   # fixed cross-backend Lanczos start seed for §20 (mulberry32+gauss; matches server/index.html)


def _lanczos_core_q0(hvp, p, m, device, dtype, q0):
    """k≤m steps of Lanczos with twice-iterated full reorthogonalisation, started from q0 (NOT torch-RNG).
    Returns (Q basis list, tridiagonal T (k×k, float64 CPU), k). MIRRORS server._lanczos_core(..., q0=q0)."""
    q = q0.to(device=device, dtype=dtype)
    q = q / q.norm()
    Q, al, be = [], [], []
    qp = torch.zeros(p, dtype=dtype, device=device)
    beta = torch.zeros((), dtype=dtype, device=device)
    for _ in range(min(p, m)):
        Q.append(q)
        w = hvp(q)
        a = w @ q
        al.append(float(a))
        w = w - a * q - beta * qp
        Qm = torch.stack(Q)                        # (k, p)
        for _ in range(2):                         # twice-iterated classical Gram–Schmidt (stable)
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
        T[i, i + 1] = be[i]
        T[i + 1, i] = be[i]
    return Q, T, k


def sec20_payload(model, Jc, rr, th, X, N, outD, K, mr_hvp=None):
    """§20 record dict {lam, q1, q2, q3, q4, K}. MIRRORS server._sec20_payload EXACTLY.

    model : eos_lab Model (used for the M_r operator hvp_S; not used when mr_hvp is supplied)
    Jc    : per-output Jacobian (M_full, p) from jac_cols(model, th, X)[0]  (M = N·outD rows used)
    rr    : generic residual (Y−f for MSE) flat (M_full,)  [= (-N·loss.resid_cotangent(out,Y,N)).reshape(-1)]
    th    : current θ (p,) ;  X : data ;  N : #samples ;  outD : output dim ;  K : eigenpairs per side
    mr_hvp: optional surrogate M_r operator v↦M_r·v (§22/§23). Default = hvp_S(model, th, X, rc, v).
    """
    M = N * outD
    Jg = Jc[:M]                                    # M×p effective-sample Jacobian
    r = rr[:M]                                      # M residual (Y−f for MSE, onehot−softmax for CE)
    p = Jg.shape[1]
    device, dtype = Jg.device, Jg.dtype
    rc = r.reshape(N, outD)
    Klan = max(1, min(int(K), p))
    mlan = min(p, max(5 * Klan, 64))                # 5K: converge the top-K⊕bottom-K eigenpairs (2K+16 under-converges)
    q0 = _randvec16(p, SEC20_SEED, device, dtype)   # mulberry32+gauss start (cross-backend identical)
    # §22/§23 pass a surrogate M_r operator; §20 default = Σ_k r_k ∇²f_k at θ via hvp_S (note: eos_lab
    # hvp_S signature is hvp_S(model, theta, X, c, v) — c (cotangent rc) BEFORE v).
    mhvp = mr_hvp if mr_hvp is not None else (lambda v: hvp_S(model, th, X, rc, v))
    Q, T, k = _lanczos_core_q0(mhvp, p, mlan, device, dtype, q0)
    mu, Sv = safe_eigh(T)                           # k Ritz pairs of M_r, ascending
    desc = torch.argsort(mu, descending=True)       # indices, descending eigenvalue
    Qmat = torch.stack(Q)                           # (k, p)

    def ritz(ix):                                   # Ritz vector in θ-space (p,)
        return Sv[:, ix].to(device=device, dtype=dtype) @ Qmat

    kt = min(Klan, k)
    bs = max(kt, k - Klan)                           # top-K ⊕ bottom-K DISTINCT positions (no double-count when 2K>k)
    positions = list(range(kt)) + list(range(bs, k))
    Kc, Vc = sym_eig_desc(Jg @ Jg.t())              # M×M NTK, eigvecs in columns (descending eigenvalue σ_k²)
    ntol = 1e-9 * max(float(Kc[0]) if int(Kc.numel()) > 0 else 0.0, 1e-30)   # null-direction cutoff σ_k²≤tol·σ_1²
    zero = torch.zeros(p, dtype=dtype, device=device)
    us = []
    for kk in range(4):                             # top-4 right-singular vectors u_k = normalize(Jᵀ g_k); GATE on the
        if kk < M and float(Kc[kk]) > ntol:         #   NTK eigenvalue, not ‖u_k‖ (a zero σ_k² ⇒ Jᵀg_k ≈ round-off noise)
            uk = Jg.t() @ Vc[:, kk]
            nu = float(uk.norm())
            us.append(uk / nu if nu > 1e-30 else zero)
        else:
            us.append(zero)
    lam, q1, q2, q3, q4 = [], [], [], [], []
    scN = 1.0 / max(N, 1)                            # ÷N: report (1/N)Σ_k r_k Q_k — comparable to §1/§2/§3 (also ÷N)
    for pos in positions:
        ix = int(desc[pos])
        li = float(mu[ix]) * scN
        vi = ritz(ix)
        lam.append(li)
        q1.append(li * abs(float(vi @ us[0])))
        q2.append(li * abs(float(vi @ us[1])))
        q3.append(li * abs(float(vi @ us[2])))
        q4.append(li * abs(float(vi @ us[3])))
    return {"lam": lam, "q1": q1, "q2": q2, "q3": q3, "q4": q4, "K": Klan}
