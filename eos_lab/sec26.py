"""§26: eigenvector-direction drift — |cos(v_i(t), v_i(t−k))|, k∈{10,20,30,50,100}, for the top-3 eigenvectors of
the Gauss–Newton GN=JᵀJ (p-dim) and the NTK=JJᵀ (M-dim), and the top-3 ⊕ bottom-3 eigenvectors of the
residual-contracted function Hessian M_r=Σ_k r_kQ_k (p-dim). |cos| because eigvec signs are arbitrary.

MIRRORS server._sec26_eigvecs / server._sec26_drift EXACTLY. GN & NTK top-3 come from ONE eigendecomposition
of the M×M NTK (NTK eigvec = g_k; GN eigvec = normalize(Jᵀ g_k)). M_r top-3⊕bottom-3 via §20's byte-parity
matrix-free Lanczos on hvp_S (SEC20_SEED q0, _lanczos_core_q0) ⇒ the M_r eigvecs match server to Lanczos
tolerance. Since every drift is |cos|, per-eigenvector sign flips across backends are gauge-free.
"""
import torch

from .models import hvp_S
from .linalg import safe_eigh, sym_eig_desc
from .sec16 import _randvec16
from .sec20 import SEC20_SEED, _lanczos_core_q0


def sec26_eigvecs(model, Jc, rr, th, X, N, outD, mr_hvp=None):
    """Return {'gn','ntk','mrTop','mrBot'} — each a list of 3 unit vectors (zero for a null/negligible direction):
      gn[i]    = i-th TOP eigenvector of GN=JᵀJ (p-dim) = normalize(Jᵀ g_i)
      ntk[i]   = i-th TOP eigenvector of NTK=JJᵀ (M-dim) = g_i (left-singular vector of J)
      mrTop[i] = i-th TOP eigenvector of M_r=Σ_k r_kQ_k (p-dim);  mrBot[i] = i-th BOTTOM (bot[0]=most negative)
    An M_r eigenvector is NULL (→ drift None) when |λ_i(M_r)| ≤ 1e-6·λ_max(NTK): once r→0 (loss converges) M_r→0 and
    its eigen-directions are ill-defined ⇒ blank rather than plot noise. MIRRORS server._sec26_eigvecs line-by-line."""
    M = N * outD
    Jg = Jc[:M]; r = rr[:M]; p = Jg.shape[1]
    device, dtype = Jg.device, Jg.dtype
    rc = r.reshape(N, outD)
    zP = torch.zeros(p, dtype=dtype, device=device); zM = torch.zeros(M, dtype=dtype, device=device)
    # --- GN & NTK top-3 from ONE eigendecomposition of the M×M NTK ---
    Kc, Vc = sym_eig_desc(Jg @ Jg.t())                     # NTK eigvecs (columns, descending σ²)
    ntop = float(Kc[0]) if int(Kc.numel()) > 0 else 0.0
    ntol = 1e-9 * max(ntop, 1e-30)                         # null-direction cutoff (GN/NTK)
    ntk, gn = [], []
    for kk in range(3):
        if kk < M and float(Kc[kk]) > ntol:
            gk = Vc[:, kk].clone()                         # NTK eigvec (M-dim), unit
            uk = Jg.t() @ gk; nu = float(uk.norm())        # GN eigvec (p-dim) = normalize(Jᵀ g_k)
            ntk.append(gk); gn.append(uk / nu if nu > 1e-30 else zP.clone())
        else:
            ntk.append(zM.clone()); gn.append(zP.clone())
    # --- M_r=Σ_k r_kQ_k top-3 ⊕ bottom-3 via §20's Lanczos on hvp_S (same q0/start) ---
    Klan = 3; mlan = min(p, max(5 * Klan, 64))
    q0 = _randvec16(p, SEC20_SEED, device, dtype)
    mhvp = mr_hvp if mr_hvp is not None else (lambda v: hvp_S(model, th, X, rc, v))   # eos_lab hvp_S(model,θ,X,c,v): c=rc BEFORE v
    Q, T, k = _lanczos_core_q0(mhvp, p, mlan, device, dtype, q0)
    mu, Sv = safe_eigh(T); desc = torch.argsort(mu, descending=True); Qmat = torch.stack(Q)
    mtol = 1e-6 * max(ntop, 1e-30)                         # blank M_r eigvecs with |λ| below this (M_r≈0 ⇒ noise direction)
    def ritz(pos):                                         # unit Ritz vector for desc[pos]; null if that M_r eigenvalue is negligible
        ix = int(desc[pos])
        if abs(float(mu[ix])) <= mtol: return zP.clone()
        v = Sv[:, ix].to(device=device, dtype=dtype) @ Qmat; nv = float(v.norm())
        return v / nv if nv > 1e-30 else zP.clone()
    kt = min(3, k)
    mrTop = [ritz(i) for i in range(kt)]                             # largest eigenvalues
    mrBot = [ritz(k - 1 - i) for i in range(3) if (k - 1 - i) >= kt]  # most-negative first, DISTINCT from the top-kt (no double-count when k<6)
    while len(mrTop) < 3: mrTop.append(zP.clone())
    while len(mrBot) < 3: mrBot.append(zP.clone())
    return {"gn": gn, "ntk": ntk, "mrTop": mrTop, "mrBot": mrBot}


def sec26_drift(cur, hist, t):
    """g26 = {key: [[|cos(v_i(t),v_i(t−k))| for k∈{10,20,30,50,100}] for i in 0..2]} for key in gn/ntk/mrTop/mrBot.
    None where that lag isn't in history yet or a vector is null. |cos| ⇒ eigvec sign flips are gauge-free.
    MIRRORS server._sec26_drift."""
    byT = {e["t"]: e for e in hist}
    lags = (10, 20, 30, 50, 100)
    out = {}
    for key in ("gn", "ntk", "mrTop", "mrBot"):
        rows = []
        for i in range(3):
            a = cur[key][i]; na = float(a.norm()); row = []
            for kk in lags:
                prev = byT.get(t - kk)
                if prev is None or na <= 1e-30:
                    row.append(None); continue
                b = prev[key][i]; nb = float(b.norm())
                row.append(abs(float(a @ b)) / (na * nb) if nb > 1e-30 else None)
            rows.append(row)
        out[key] = rows
    return out
