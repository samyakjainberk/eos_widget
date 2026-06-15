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


def sec12_payload(Qmats, r, energyfrac):
    """§12 per-sample Hessian eigenvector cross-similarity. Qmats:(N,p,p), r:(N,), energyfrac∈(0,1].
    Returns {M, p1, p2, p3} — see server._sec12_payload (this MIRRORS it exactly).
      p1/p2 (k=2 / k=5): {g:[grid1,grid3,grid5] (N·N), c:[cub2,cub4,cub6] (N·N·10), gm,cm,cs, K}.
      p3 (principal angles, k=1,5,10,energy): {g:[4 grids N·N], gm:[4 means in degrees]}.
    grid1=max_{a,b}|cos(e_i^a,e_j^b)|; cub2=top-10 |cos|; grid3=signed cos of argmax × sign(λ_i)·sign(λ_j);
    cub4=same for top-10; grid5/cub6 = ×sign(r_i)·sign(r_j); panel-3 cell = MEAN principal angle of the
    2k (energy=|λ|-energyfrac) eigen-subspaces of Q_i and Q_j."""
    N, p = int(Qmats.shape[0]), int(Qmats.shape[1])
    dev, dt = Qmats.device, Qmats.dtype
    Vd = torch.empty(N, p, p, dtype=dt, device=dev)     # eigenvectors columns DESCENDING by eigenvalue
    wd = torch.empty(N, p, dtype=dt, device=dev)
    for i in range(N):
        w, V = torch.linalg.eigh(0.5 * (Qmats[i] + Qmats[i].t()))
        wd[i] = w.flip(0); Vi = V.flip(1)
        mx = Vi.abs().argmax(dim=0)                      # pin sign: largest-|component| positive (reproducible)
        sg = torch.sign(Vi[mx, torch.arange(p, device=dev)])
        Vd[i] = Vi * torch.where(sg == 0, torch.ones_like(sg), sg).view(1, p)
    rs = torch.sign(r).to(dt)

    def vecset(k):
        kk = max(1, min(k, p // 2))
        V2 = torch.cat([Vd[:, :, :kk], Vd[:, :, p - kk:]], dim=2)
        wv = torch.cat([wd[:, :kk], wd[:, p - kk:]], dim=1)
        return V2, torch.sign(wv).to(dt)

    def cos_panel(k):
        E, s = vecset(k)
        D = E.shape[2]
        C = torch.einsum('ipa,jpb->ijab', E, E)
        Cf = C.reshape(N, N, D * D)
        Sab = torch.einsum('ia,jb->ijab', s, s).reshape(N, N, D * D)
        Mf = Cf * Sab
        absf = Cf.abs()
        sr = rs.view(N, 1) * rs.view(1, N)
        g1 = absf.amax(dim=2)
        g3 = torch.gather(Mf, 2, absf.argmax(dim=2, keepdim=True)).squeeze(2)
        g5 = g3 * sr
        K = min(10, D * D)
        tv, ti = torch.topk(absf, K, dim=2)
        c2 = tv; c4 = torch.gather(Mf, 2, ti); c6 = c4 * sr.unsqueeze(2)
        grids, cubs = [g1, g3, g5], [c2, c4, c6]
        return {"g": [x.reshape(-1).detach().cpu().tolist() for x in grids],
                "c": [x.reshape(-1).detach().cpu().tolist() for x in cubs],
                "gm": [float(x.mean()) for x in grids],
                "cm": [float(x.mean()) for x in cubs],
                "cs": [float(x.std(unbiased=False)) for x in cubs], "K": K}

    def pa_fixed(k):
        E, _ = vecset(k)
        sv = torch.linalg.svdvals(torch.einsum('ipa,jpb->ijab', E, E)).clamp(-1.0, 1.0)
        return (torch.acos(sv) * (180.0 / math.pi)).mean(dim=2)

    def pa_energy():
        Es = []
        for i in range(N):
            aw = wd[i].abs()
            order = torch.argsort(aw, descending=True)
            cw = torch.cumsum(aw[order], dim=0)
            tot = float(cw[-1]) if float(cw[-1]) > 0 else 1.0
            d = max(1, min(int((cw < energyfrac * tot).sum().item()) + 1, p))
            Es.append(Vd[i][:, order[:d]])
        G = torch.zeros(N, N, dtype=dt, device=dev)
        for i in range(N):
            for j in range(N):
                sv = torch.linalg.svdvals(Es[i].t() @ Es[j]).clamp(-1.0, 1.0)
                G[i, j] = (torch.acos(sv) * (180.0 / math.pi)).mean()
        return G

    pas = [pa_fixed(1), pa_fixed(5), pa_fixed(10), pa_energy()]
    return {"M": N, "p1": cos_panel(2), "p2": cos_panel(5),
            "p3": {"g": [x.reshape(-1).detach().cpu().tolist() for x in pas],
                   "gm": [float(x.mean()) for x in pas]}}


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
