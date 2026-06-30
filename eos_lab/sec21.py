"""§21 — residual ↔ spectrum alignment (THREE panels), MIRRORS server.py `_sec21_payload`.

Per-step diagnostic (cube-cadence in the integration). Given the per-output Jacobian J=Jc[:M]
(rows ∇f_k, M=N·outD) and the generic residual r=rr[:M] (Y−f for MSE, onehot−softmax for CE),
emit three 4-bar panels (grey eigenvalue bar only on the eigval-weighted #3/#4):

  Panel 1 (NTK):  K = J Jᵀ  (M×M) eigenpairs (σ_i, v_i, descending), σ_i ÷N (#samples, as G=(1/N)JᵀJ).
      Project the residual r:  n1=|⟨v_i,r⟩|/‖r‖, n2=|⟨v_i,r⟩|, n3=σ_i·n1, n4=σ_i·n2 ; grey = σ_i. (top-min(K,M))
  Panel 2 (M_r): top-K ⊕ bottom-K eigenpairs (λ_i, u_i) of M_r=Σ_k r_k Q_k (Q_k=∇²f_k), λ_i ÷N (as §20) —
      matrix-free Lanczos on the residual-weighted function-Hessian HVP started from the cross-backend
      mulberry32 vector (SEC21_SEED = SEC20_SEED ⇒ M_r eigenpairs IDENTICAL to §20). Project J·r=Σ_k r_k∇f_k:
      p1=|⟨u_i,Jr⟩|/‖Jr‖, p2=|⟨u_i,Jr⟩|, p3=λ_i·p1, p4=λ_i·p2 ; grey = signed λ_i.
  Panel 3 (Gauss-Newton): G=(1/N)JᵀJ shares its NONZERO spectrum with the NTK; for the i-th NTK mode
      (v_i, s_i²=Kc[i]) the Gauss-Newton eigenvector is the right singular vector z_i=Jᵀv_i/‖·‖ (p-dim),
      eigenvalue c_i=s_i²/N (= the ÷N NTK σ_i). rank(G)≤M ⇒ min(K,M) nonzero modes. Project J·r:
      gp1=|⟨z_i,Jr⟩|/‖Jr‖, gp2=|⟨z_i,Jr⟩|, gp3=c_i·gp1, gp4=c_i·gp2 ; grey = c_i (≥0).

Record (parity-identical to server `_sec21_payload`):
  {"sig","n1","n2","n3","n4", "lam","p1","p2","p3","p4", "cg","gp1","gp2","gp3","gp4",
   "K":Klan, "ntk":nt, "gn":ng}

EXACT-AUTOGRAD vs server FD: the NTK/Gauss-Newton parts (Panels 1/3) use only J (jac_cols, exact vjp) →
match server to ~1e-12. The M_r Lanczos (Panel 2) runs on hvp_S (eos_lab exact ∇²f·v) vs server's FD
hvpS, BOTH started from the IDENTICAL mulberry32 q0 — eigenpairs match to Lanczos convergence (~1e-6..1e-10);
the small residual is the FD-vs-exact gap of the M_r operator, NOT a port bug.
"""
import torch

from .models import jac_cols, hvp_S
from .linalg import sym_eig_desc, safe_eigh
from .sec16 import _randvec16   # mulberry32+gauss start vector — byte-identical to server._randvec16


SEC21_SEED = 0x205EE0   # = server.SEC20_SEED = SEC21_SEED: §21 Panel-2 M_r Lanczos start (cross-backend identical)


def _lanczos_core_q0(hvp, p, m, q0, dev, dt):
    """k≤m Lanczos steps (twice-iterated full reorthogonalisation) started from a GIVEN q0.
    Returns (Q list, tridiagonal T (k×k float64 CPU), k). MIRRORS server._lanczos_core with q0 set
    (same as linalg._lanczos_core but using q0 instead of the torch-RNG randn_vec start)."""
    q = q0.to(dtype=dt, device=dev)
    q = q / q.norm()
    Q, al, be = [], [], []
    qp = torch.zeros(p, dtype=dt, device=dev)
    beta = torch.zeros((), dtype=dt, device=dev)
    for _ in range(min(p, m)):
        Q.append(q)
        w = hvp(q)
        a = w @ q
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
        T[i, i + 1] = be[i]
        T[i + 1, i] = be[i]
    return Q, T, k


def sec21_payload(model, Jc, rr, th, X, N, outD, K):
    """§21 record (3 panels). MIRRORS server._sec21_payload EXACTLY.
    model: eos_lab model (for the M_r=Σ_k r_k ∇²f_k HVP via hvp_S); Jc: (M,p) per-output Jacobian from
    jac_cols; rr: (M,) generic residual; th: current θ; X: inputs; N: #input samples; outD: output dim;
    K: requested rank (s29k)."""
    M = N * outD
    Jg = Jc[:M]                                            # M×p effective-sample Jacobian (rows ∇f_k)
    r = rr[:M]                                             # M residual
    p = Jg.shape[1]
    dev, dt = Jg.device, Jg.dtype
    rc = r.reshape(N, outD)
    scN = 1.0 / max(N, 1)                                  # NTK eigvals ÷ #samples (matches G=(1/N)JᵀJ and Panel-2 M_r ÷N)

    # ---- Panel 1: NTK = Jg Jgᵀ, residual projected onto top eigenvectors ----
    Kc, Vc = sym_eig_desc(Jg @ Jg.t())                    # M×M NTK, eigvals desc (σ_i), eigvecs in columns (v_i, M-dim)
    rn = max(float(r.norm()), 1e-30)
    nt = min(int(K), M)
    sig, n1, n2, n3, n4 = [], [], [], [], []
    for i in range(nt):
        si = float(Kc[i]) * scN
        ap = abs(float(Vc[:, i] @ r))
        sig.append(si)
        n1.append(ap / rn)
        n2.append(ap)
        n3.append(si * ap / rn)
        n4.append(si * ap)

    # ---- Panel 2: M_r=Σ_k r_kQ_k (÷N) top-K⊕bottom-K, J·r=Jgᵀr projected onto eigenvectors ----
    Klan = max(1, min(int(K), p))
    mlan = min(p, max(5 * Klan, 64))   # 5K: converge the top-K⊕bottom-K eigenpairs (2K+16 under-converged the K-th for large p)
    q0 = _randvec16(p, SEC21_SEED, dev, dt)
    # server: hvpS(th, X, v, rc) = (Σ_a rc_a ∇²f_a)·v  ↔  eos_lab hvp_S(model, th, X, rc, v) (c=rc before v)
    Q, T, k = _lanczos_core_q0(lambda v: hvp_S(model, th, X, rc, v), p, mlan, q0, dev, dt)
    mu, Sv = safe_eigh(T)
    desc = torch.argsort(mu, descending=True)
    Qmat = torch.stack(Q)

    def ritz(ix):
        return Sv[:, ix].to(device=dev, dtype=dt) @ Qmat

    kt = min(Klan, k)
    bs = max(kt, k - Klan)
    positions = list(range(kt)) + list(range(bs, k))      # top-K ⊕ bottom-K DISTINCT
    Jr = Jg.t() @ r                                       # J·r = Σ_k r_k ∇f_k (p,)
    Jrn = max(float(Jr.norm()), 1e-30)
    lam, p1, p2, p3, p4 = [], [], [], [], []
    for pos in positions:
        ix = int(desc[pos])
        li = float(mu[ix]) * scN
        ui = ritz(ix)
        ap = abs(float(ui @ Jr))
        lam.append(li)
        p1.append(ap / Jrn)
        p2.append(ap)
        p3.append(li * ap / Jrn)
        p4.append(li * ap)

    # ---- Panel 3: Gauss-Newton G=(1/N)JᵀJ eigenpairs (z_i, c_i), J·r projected onto its TOP eigenvectors ----
    ng = min(int(K), M)
    cg, gp1, gp2, gp3, gp4 = [], [], [], [], []
    ntol = 1e-9 * float(Kc[0]) if Kc.numel() else 0.0        # gate on the EIGENVALUE like §20, not ‖z‖ (a ~0 NTK eigval ⇒ z_i is round-off, backend-divergent if normalized)
    for i in range(ng):
        zi = Jg.t() @ Vc[:, i]                                # z_i ∝ Jgᵀ v_i (p,)
        zn = float(zi.norm())
        if float(Kc[i]) <= ntol or zn < 1e-30:                # null-space / rank-deficient mode (J·r ⟂ it ⇒ true projection 0)
            cg.append(0.0)
            gp1.append(0.0)
            gp2.append(0.0)
            gp3.append(0.0)
            gp4.append(0.0)
            continue
        zi = zi / zn
        ci = float(Kc[i]) * scN                               # Gauss-Newton eigenvalue c_i = s_i²/N
        ap = abs(float(zi @ Jr))
        cg.append(ci)
        gp1.append(ap / Jrn)
        gp2.append(ap)
        gp3.append(ci * ap / Jrn)
        gp4.append(ci * ap)

    return {"sig": sig, "n1": n1, "n2": n2, "n3": n3, "n4": n4,
            "lam": lam, "p1": p1, "p2": p2, "p3": p3, "p4": p4,
            "cg": cg, "gp1": gp1, "gp2": gp2, "gp3": gp3, "gp4": gp4,
            "K": Klan, "ntk": nt, "gn": ng}
