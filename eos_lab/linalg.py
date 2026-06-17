"""
Matrix-free eigen-analysis — Lanczos, SLQ, principal angles, energy subspaces.

MIRRORS:  server.py  (_randn_vec, _lanczos_core, lanczos_extreme[_vals], slq_density,
                       sym_eig_desc, pos_subspace/neg_subspace, principal_angles, sign_to, pin_sign)
          ↔ index.html  (the same Lanczos / SLQ / rotation panels, §1–§3, §5, §6)

Everything here takes an `hvp` CLOSURE  v -> H·v  (built in models.py / diagnostics.py) and never
forms a p×p matrix — so it scales to the 10⁷-parameter cap. The tridiagonal T is diagonalised on
CPU/float64 for stability; Ritz vectors are reconstructed on the working device.
"""
import math
import torch


def randn_vec(p, seed, device, dtype):
    """Seeded standard-normal probe vector (p,). MIRRORS server._randn_vec."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) & 0x7FFFFFFF)
    return torch.randn(p, dtype=dtype, device=device, generator=g)


def _lanczos_core(hvp, p, m, seed, device, dtype):
    """k≤m steps of Lanczos with twice-iterated full reorthogonalisation.
    Returns (Q basis list, tridiagonal T (k×k, float64 CPU), k). MIRRORS server._lanczos_core."""
    q = randn_vec(p, seed, device, dtype)
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
        Qm = torch.stack(Q)                       # (k, p)
        for _ in range(2):                        # twice-iterated classical Gram–Schmidt (stable)
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


def _eig_sorted(T):
    mu, Sv = torch.linalg.eigh(T)                 # ascending, CPU float64
    idx = torch.argsort(mu, descending=True)
    return mu, Sv, idx


def sym_eig_desc(A):
    """Eigen-decomposition of a small symmetric matrix, descending. Returns (vals, vecs cols).
    MIRRORS server.sym_eig_desc (used by the multi-sample §7/§8/§4b–d/§9 panels)."""
    w, V = torch.linalg.eigh(A)
    idx = torch.argsort(w, descending=True)
    return w[idx], V[:, idx]


def lanczos_extreme(hvp, p, n, m, seed, device, dtype):
    """Top-n and bottom-n (eigenvalue, eigenVECTOR) pairs. MIRRORS server.lanczos_extreme.
    Returns (top_vecs, bot_vecs, top_vals, bot_vals)."""
    Q, T, k = _lanczos_core(hvp, p, m, seed, device, dtype)
    mu, Sv, idx = _eig_sorted(T)
    Qmat = torch.stack(Q)                         # (k, p)

    def ritz(col):
        c = Sv[:, col].to(device=device, dtype=dtype)
        return c @ Qmat                           # (p,)

    top_vecs = [ritz(int(idx[min(i, k - 1)])) for i in range(n)]
    bot_vecs = [ritz(int(idx[max(0, k - 1 - i)])) for i in range(n)]
    top_vals = [float(mu[int(idx[min(i, k - 1)])]) for i in range(n)]
    bot_vals = [float(mu[int(idx[max(0, k - 1 - i)])]) for i in range(n)]
    return top_vecs, bot_vecs, top_vals, bot_vals


def batched_lanczos_extreme(hvp_batch, p, N, n, m, seeds, device, dtype):
    """N independent Lanczos eigendecompositions IN LOCKSTEP, vectorised over the N samples. hvp_batch(V:(N,p))
    → (N,p) = {Q_i·V_i}_i (one batched HVP per step). Returns top-n & bottom-n eigenpairs per sample:
    (TV,TW,BV,BW) = (N,n,p),(N,n),(N,n,p),(N,n). MIRRORS server.batched_lanczos_extreme (used by §12)."""
    q = torch.stack([randn_vec(p, seeds[i], device, dtype) for i in range(N)])
    q = q / q.norm(dim=1, keepdim=True)
    Qs, al, be = [], [], []
    qp = torch.zeros(N, p, dtype=dtype, device=device); beta = torch.zeros(N, dtype=dtype, device=device)
    done = torch.zeros(N, dtype=torch.bool, device=device)          # Krylov-exhausted samples (freeze, q=0)
    bmax = torch.zeros(N, dtype=dtype, device=device)
    for _ in range(min(p, m)):
        Qs.append(q)
        w = hvp_batch(q)
        a = (w * q).sum(1); al.append(a)
        w = w - a[:, None] * q - beta[:, None] * qp
        Qm = torch.stack(Qs, dim=1)
        for _ in range(2):
            w = w - torch.bmm(Qm.transpose(1, 2), torch.bmm(Qm, w.unsqueeze(2))).squeeze(2)
        beta = w.norm(dim=1)
        bmax = torch.maximum(bmax, beta)
        done = done | (beta < 1e-6 * bmax.clamp_min(1e-30) + 1e-30)  # breakdown ⇒ freeze (no noise vector in the basis)
        beta = torch.where(done, torch.zeros_like(beta), beta)
        be.append(beta); qp = q
        q = torch.where(done.unsqueeze(1), torch.zeros_like(w), w / beta.clamp_min(1e-30).unsqueeze(1))
    Qmat = torch.stack(Qs, dim=1); k = len(Qs)
    T = torch.zeros(N, k, k, dtype=dtype, device=device)
    for i in range(k):
        T[:, i, i] = al[i]
    for i in range(k - 1):
        T[:, i, i + 1] = be[i]; T[:, i + 1, i] = be[i]
    mu, Sv = torch.linalg.eigh(T)
    nn = min(n, k)
    top = mu.argsort(dim=1, descending=True)[:, :nn]; bot = mu.argsort(dim=1)[:, :nn]

    def gp(ix):
        vals = torch.gather(mu, 1, ix)
        Svs = torch.gather(Sv, 2, ix.unsqueeze(1).expand(N, k, nn))
        vecs = torch.einsum('nkp,nkj->njp', Qmat, Svs)
        vecs = vecs / vecs.norm(dim=2, keepdim=True).clamp_min(1.0)  # safety: shrink any |·|>1 Ritz vector to unit ⇒ |cos|≤1
        return vecs, vals
    tv, tw = gp(top); bv, bw = gp(bot)
    return tv, tw, bv, bw


def lanczos_extreme_vals(hvp, p, n, m, seed, device, dtype):
    """Top-n and bottom-n eigenVALUES only (cheaper — no Ritz reconstruction).
    MIRRORS server.lanczos_extreme_vals. Returns (top, bot)."""
    _, T, k = _lanczos_core(hvp, p, m, seed, device, dtype)
    mu, _, idx = _eig_sorted(T)
    top = [float(mu[int(idx[min(i, k - 1)])]) for i in range(n)]
    bot = [float(mu[int(idx[max(0, k - 1 - i)])]) for i in range(n)]
    return top, bot


def hutch_trace(hvp, p, nprobe, seed, device, dtype):
    """Hutchinson trace estimate: trace(A) ≈ (1/k) Σ zᵀ(A z), z Rademacher ±1. MIRRORS server._hutch_trace."""
    s = 0.0
    for i in range(nprobe):
        z = torch.sign(randn_vec(p, (seed + i * 7919) & 0x7FFFFFFF, device, dtype))
        s += float(z @ hvp(z))
    return s / max(1, nprobe)


SEC12_KFULL = 15   # rank of the §12 "full-space" principal-angle plot (top-15 ⊕ bottom-15) — MIRRORS server.


def sec12_payload(TV, TW, BV, BW, r, Jg, grid3dcap, kfull=SEC12_KFULL):
    """§12 per-sample Hessian-eigenvector principal angles. MIRRORS server._sec12_payload. TV,BV:(N,K,p)
    eigenVECTORS; TW,BW:(N,K) eigenVALUES; r:(N,) residual; Jg:(N,p) per-sample GT-label gradients ∇f_i.
    Computes only the panel-4 principal angles (svdvals is gauge-free, invariant to eigenvector ±). The old
    §12 cube grids / §13 g1,g2,g3 are gone (see sec13_stats). Returns {M,do3d,ks,ang}; ang = angles over all N."""
    N, K, p = int(TV.shape[0]), int(TV.shape[1]), int(TV.shape[2])
    dev = TV.device

    def pa(k):
        kk = max(1, min(k, K))
        E = torch.cat([TV[:, :kk, :], BV[:, :kk, :]], dim=1)
        sv = torch.linalg.svdvals(torch.einsum('iap,jbp->ijab', E, E)).clamp(-1.0, 1.0)
        return (torch.acos(sv) * (180.0 / math.pi)).mean(dim=2)
    kf = max(1, min(kfull, K))
    pas = [pa(1), pa(5), pa(10), pa(kf)]
    out = {"M": N, "do3d": False, "ks": [1, 2, 5],
           "ang": {"g": [x.reshape(-1).detach().cpu().tolist() for x in pas],
                   "gm": [float(x.mean()) for x in pas], "kfull": kf}}
    return out


SEC13_KS = (2, 5)   # §13 top⊕bottom ranks k₀ compared against the exact reference
SEC13_DOMS = ("ndist", "ijeq", "diag")   # i≠j≠k ; i=j≠k ; i=j=k  (the three (i,j,k) classes plotted)


def sec13_stats(cur, prev, ref, lr):
    """§13 — approximations G1/G2/G3 of the exact reference J_iᵀQ_jJ_k·r_k, from per-sample Lanczos eigenpairs.
    For triple (i,j,k) and rank k₀ (D=2k₀ top⊕bottom dirs x,y,z of Q_i,Q_j,Q_k):
      G1 = Σ_{x,y,z} r_{k,t}σ_{j,y}·(1+r_{i,t-1}ησ_{i,x})cos_{(i,x)(j,y)}J'(x)_{i,t-1}·(1+r_{k,t-1}ησ_{k,z})cos_{(j,y)(k,z)}J'(z)_{k,t-1}
      G2 = Σ_{y,z}   r_{k,t}σ_{j,y}·(J_{i,t}·u_{j,y})·(1+r_{k,t-1}ησ_{k,z})cos_{(j,y)(k,z)}J'(z)_{k,t-1}      (i-side → J_{i,t}·u_{j,y})
      G3 = Σ_{x,y}   r_{k,t}σ_{j,y}·(1+r_{i,t-1}ησ_{i,x})cos_{(i,x)(j,y)}J'(x)_{i,t-1}·(J_{k,t}·u_{j,y})       (k-side → J_{k,t}·u_{j,y})
    Q_j (σ_{j,y}, u_{j,y}) is taken at step t; Q_i and Q_k (σ, u, J', r) at step t-1; r_{k,t} & σ_{j,y} at t. cos_{(a)(b)}
    are signed dot products of unit eigenvectors (cross-time for i↔j and j↔k). J'(x)_{i,t-1}=⟨J_{i,t-1},u_{i,x}^{t-1}⟩.
    cur/prev = {TV,TW,BV,BW,r,Jg}. ref=(N,N,N) exact J_iᵀQ_jJ_k·r_k at t. Returns mean/std over ALL (i,j,k) and over
    i≠j≠k for ref and each (G,k₀). MIRRORS server._sec13_stats / index.html sec13Stats."""
    TV, TW, BV, BW = cur["TV"], cur["TW"], cur["BV"], cur["BW"]
    pTV, pTW, pBV, pBW = prev["TV"], prev["TW"], prev["BV"], prev["BW"]
    N, K = int(TV.shape[0]), int(TV.shape[1]); dev = TV.device
    rt, Jt, pr, pJ = cur["r"], cur["Jg"], prev["r"], prev["Jg"]
    ai = torch.arange(N, device=dev)
    I, Jx, Kx = ai.view(N, 1, 1), ai.view(1, N, 1), ai.view(1, 1, N)
    masks = {"ndist": ((I != Jx) & (Jx != Kx) & (I != Kx)).reshape(-1),   # i≠j≠k (all distinct)
             "ijeq":  ((I == Jx) & (Jx != Kx)).reshape(-1),               # i=j≠k
             "diag":  ((I == Jx) & (Jx == Kx)).reshape(-1)}               # i=j=k (self-triple diagonal)

    def stats(T):
        f = T.reshape(-1)
        return {nm: ([float(f[mk].mean()), float(f[mk].std(unbiased=False))] if int(mk.sum()) else [0.0, 0.0])
                for nm, mk in masks.items()}

    out = {"ref": stats(ref), "g1": {}, "g2": {}, "g3": {}}
    for k0 in SEC13_KS:
        kk = max(1, min(k0, K))
        Uj = torch.cat([TV[:, :kk], BV[:, :kk]], dim=1)              # (N,D,p) eigvecs of Q_j at t
        Wj = torch.cat([TW[:, :kk], BW[:, :kk]], dim=1)              # (N,D) eigvals σ_{j,y} at t
        Up = torch.cat([pTV[:, :kk], pBV[:, :kk]], dim=1)            # (N,D,p) eigvecs of Q_i/Q_k at t-1
        Wp = torch.cat([pTW[:, :kk], pBW[:, :kk]], dim=1)            # (N,D) eigvals at t-1
        Jp = torch.einsum('ip,iap->ia', pJ, Up)                     # J'(x)_{i,t-1} = ⟨J_{t-1},u^{t-1}⟩
        Cij = torch.einsum('ixp,jyp->ijxy', Up, Uj)                 # cos_{(i,x)(j,y)} = u_i^{t-1}·u_j^{t}
        Cjk = torch.einsum('jyp,kzp->jkyz', Uj, Up)                 # cos_{(j,y)(k,z)} = u_j^{t}·u_k^{t-1}
        P = (1.0 + pr.view(N, 1) * lr * Wp) * Jp                    # (1+r_{t-1}ησ_{t-1})·J'_{t-1}  (i- & k-side)
        a = torch.einsum('ix,ijxy->ijy', P, Cij)                    # i-side sum over x
        b = torch.einsum('kz,jkyz->jky', P, Cjk)                    # k-side sum over z
        av = a * Wj.unsqueeze(0)                                    # σ_{j,y}·a
        Jiu = torch.einsum('ip,jyp->ijy', Jt, Uj)                   # J_{i,t}·u_{j,y}^{t}  (G2 i-side)
        Jku = torch.einsum('kp,jyp->jky', Jt, Uj)                   # J_{k,t}·u_{j,y}^{t}  (G3 k-side)
        Jiuv = Jiu * Wj.unsqueeze(0)
        rk = rt.view(1, 1, N)
        G1 = rk * torch.einsum('ijy,jky->ijk', av, b)
        G2 = rk * torch.einsum('ijy,jky->ijk', Jiuv, b)
        G3 = rk * torch.einsum('ijy,jky->ijk', av, Jku)
        key = str(k0)
        out["g1"][key] = stats(G1); out["g2"][key] = stats(G2); out["g3"][key] = stats(G3)
    return out


SEC14_YS = (1, 2, 5)


def sec14_payload(TV, TW, BV, BW, r, Jg, lr, grid3dcap, rhist):
    """§14 — per-triplet (i,j,k) decomposition of Tr(ΔNTK)=Σ_{ℓ,p} a·b·c·d·e (a=u_{i,ℓ}·u_{j,p}, b=σ_{i,ℓ}J_i·u_{i,ℓ},
    c=1+ησ_{j,p}r_j, d=u_{j,p}·J_k, e=r_k). Rank-2y, gradient-gauged; the (2y)² terms are sorted signed-descending
    per cell. Panels per y∈{1,2,5}: agg=[max,min,Σtop-y,Σbot-y,Σall]; max/min=[a,b·d,a·b·d,a·b·d·e,a·e] at the
    argmax/argmin. Panel 10: ∏_{z=1}^{S}(1+ησ_{j,p}r_{j,t-S+z}) (S=y) over the residual history, first/last × y.
    DENSE here (no SSE); values parity-match server._sec14_payload. MIRRORS server / index.html sec14Payload."""
    N, K, p = int(TV.shape[0]), int(TV.shape[1]), int(TV.shape[2])
    out = {"M": N, "do3d": False}
    if N > grid3dcap:
        return out
    out["do3d"] = True
    dev = TV.device
    dg = torch.arange(N, device=dev)

    def pack14(T):
        f = T.reshape(-1); pos = f[f > 0]; neg = f[f < 0]
        return {"v": f.detach().cpu().tolist(), "idx": None, "d": T[dg, dg, dg].detach().cpu().tolist(),
                "mn": float(f.mean()), "mp": float(pos.mean()) if pos.numel() else 0.0,
                "mng": float(neg.mean()) if neg.numel() else 0.0}

    def factors(y):
        kk = max(1, min(y, K)); D = 2 * kk
        E = torch.cat([TV[:, :kk, :], BV[:, :kk, :]], dim=1)
        W = torch.cat([TW[:, :kk], BW[:, :kk]], dim=1)
        gE = torch.einsum('ip,idp->id', Jg, E)
        E = E * torch.where(gE >= 0, torch.ones_like(gE), -torch.ones_like(gE)).unsqueeze(2)   # gradient gauge
        Ju = torch.einsum('ip,idp->id', Jg, E)
        a = torch.einsum('ilp,jmp->ijlm', E, E)
        b = W * Ju
        c = 1.0 + lr * W * r.view(N, 1)
        d = torch.einsum('jmp,kp->jmk', E, Jg)
        e = r
        T = (a.unsqueeze(2) * b.view(N, 1, 1, D, 1) * c.view(1, N, 1, 1, D)
             * d.permute(0, 2, 1).reshape(1, N, N, 1, D) * e.view(1, 1, N, 1, 1))
        return a, b, d, e, W, T, D, kk

    p14 = {}; sig_sel = {"first": {}, "last": {}}
    for y in SEC14_YS:
        a, b, d, e, W, T, D, kk = factors(y)
        Tf = T.reshape(N, N, N, D * D)
        Ts, _ = torch.sort(Tf, dim=3, descending=True)
        agg = [Ts[..., 0], Ts[..., -1], Ts[..., :kk].sum(3), Ts[..., D * D - kk:].sum(3), Tf.sum(3)]
        # FIRST (ℓ,p) on ties (deterministic) — matches the browser/server flat-order scan (cross-backend parity)
        ar = torch.arange(D * D, device=dev).view(1, 1, 1, D * D); big = torch.full((), D * D, device=dev, dtype=ar.dtype)
        imax = torch.where(Tf == Tf.amax(3, keepdim=True), ar, big).amin(3).clamp_(max=D * D - 1)   # clamp: NaN cell ⇒ no match ⇒ big
        imin = torch.where(Tf == Tf.amin(3, keepdim=True), ar, big).amin(3).clamp_(max=D * D - 1)

        def subprods(idx):
            lst = idx // D; mst = idx % D
            a_sel = a.reshape(N, N, D * D).unsqueeze(2).expand(N, N, N, D * D).gather(3, idx.unsqueeze(3)).squeeze(3)
            b_sel = b.view(N, 1, 1, D).expand(N, N, N, D).gather(3, lst.unsqueeze(3)).squeeze(3)
            d_sel = d.permute(0, 2, 1).reshape(1, N, N, D).expand(N, N, N, D).gather(3, mst.unsqueeze(3)).squeeze(3)
            e_sel = e.view(1, 1, N).expand(N, N, N)
            return [a_sel, b_sel * d_sel, a_sel * b_sel * d_sel, a_sel * b_sel * d_sel * e_sel, a_sel * e_sel], mst
        sub_max, mst_max = subprods(imax)
        sub_min, mst_min = subprods(imin)
        p14[str(y)] = {"agg": [pack14(x) for x in agg], "max": [pack14(x) for x in sub_max],
                       "min": [pack14(x) for x in sub_min]}
        Wj = W.view(1, N, 1, D).expand(N, N, N, D)
        sig_sel["first"][str(y)] = Wj.gather(3, mst_max.unsqueeze(3)).squeeze(3)
        sig_sel["last"][str(y)] = Wj.gather(3, mst_min.unsqueeze(3)).squeeze(3)
    out["p14"] = p14

    ratios = []
    for sel in ("first", "last"):
        for y in SEC14_YS:
            S = y
            if len(rhist) < S:
                ratios.append(None); continue
            sig = sig_sel[sel][str(y)]
            prod = torch.ones(N, N, N, device=dev, dtype=TV.dtype)
            for rt in rhist[-S:]:
                prod = prod * (1.0 + lr * sig * rt.view(1, N, 1))
            ratios.append(pack14(prod))
    out["ratio"] = ratios
    return out


def slq_density(hvp, p, nprobe, m, ngrid, seed, device, dtype):
    """Stochastic Lanczos Quadrature estimate of the spectral density (Gaussian-smoothed).
    MIRRORS server.slq_density. Returns {'x': [...], 'y': [...]} (§5)."""
    import numpy as np
    TH, W = [], []
    for pr in range(nprobe):
        _, T, k = _lanczos_core(hvp, p, min(p, m), (seed + pr * 1315423) & 0xFFFFFFFF, device, dtype)
        val, V = torch.linalg.eigh(T)
        val = val.numpy()
        v0 = V[0, :].numpy()
        for i in range(k):
            TH.append(float(val[i]))
            W.append(float(v0[i] ** 2 / nprobe))
    TH = np.asarray(TH)
    W = np.asarray(W)
    lo, hi = float(TH.min()), float(TH.max())
    if not hi > lo:
        hi, lo = lo + 1, lo - 1
    pad = 0.05 * (hi - lo) + 1e-9
    lo -= pad
    hi += pad
    sigma = max((hi - lo) / 60, 1e-9)
    inv = 1.0 / (sigma * math.sqrt(2 * math.pi))
    x = np.linspace(lo, hi, ngrid)
    y = np.array([float(np.sum(W * np.exp(-0.5 * ((lam - TH) / sigma) ** 2) * inv)) for lam in x])
    return {"x": x.tolist(), "y": y.tolist()}


# --------------------------------------------------------------------------- eigenspace rotation (§6)
def pin_sign(v):
    """Deterministic sign: make the largest-|component| positive. MIRRORS server.pin_sign."""
    mi = int(torch.argmax(torch.abs(v)))
    return v if float(v[mi]) >= 0 else -v


def sign_to(cur, prev):
    """Flip `cur` to align with `prev` (eigenvectors are sign-ambiguous). MIRRORS server.sign_to."""
    if prev is None:
        return cur
    return -cur if float(cur @ prev) < 0 else cur


def pos_subspace(vals, vecs, frac):
    """Top eigenvectors capturing `frac` of the POSITIVE eigenvalue energy. MIRRORS server.pos_subspace."""
    idx = [i for i in range(len(vals)) if vals[i] > 0]
    E = sum(vals[i] for i in idx)
    if E <= 0:
        return []
    c, sub = 0.0, []
    for i in idx:
        c += vals[i]
        sub.append(vecs[i])
        if c >= frac * E:
            break
    return sub


def neg_subspace(vals, vecs, frac):
    """Bottom eigenvectors capturing `frac` of the NEGATIVE eigenvalue energy. MIRRORS server.neg_subspace."""
    idx = [i for i in range(len(vals)) if vals[i] < 0]
    E = sum(abs(vals[i]) for i in idx)
    if E <= 0:
        return []
    c, sub = 0.0, []
    for i in idx:
        c += abs(vals[i])
        sub.append(vecs[i])
        if c >= frac * E:
            break
    return sub


def principal_angles(A, B):
    """Principal angles (degrees) between subspaces spanned by vector lists A, B.
    Returns {'mx': max angle, 'mn': mean angle} or None. MIRRORS server.principal_angles."""
    if not A or not B:
        return None
    Am = torch.stack(A)
    Bm = torch.stack(B)
    Mc = (Am @ Bm.t()).double().cpu()
    d1, d2 = Mc.shape
    G = Mc @ Mc.t() if d1 <= d2 else Mc.t() @ Mc
    vals = torch.linalg.eigvalsh(G)
    ang = []
    for v in vals:
        s = math.sqrt(max(0.0, min(1.0, float(v))))
        ang.append(math.degrees(math.acos(max(-1.0, min(1.0, s)))))
    ang.sort()
    return {"mx": ang[-1], "mn": sum(ang) / len(ang)}
