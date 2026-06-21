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
        # Real Ritz vectors have ‖·‖≈1; spurious breakdown null-space vectors ‖·‖≈0. Make it EXACT: real → unit;
        # spurious → 0 vector AND 0 eigenvalue (clean sentinel every §12/13/14 consumer excludes). MIRRORS server.
        nrm = vecs.norm(dim=2, keepdim=True)
        real = nrm > 0.5
        vecs = torch.where(real, vecs / nrm.clamp_min(1e-30), torch.zeros_like(vecs))
        vals = torch.where(real.squeeze(2), vals, torch.zeros_like(vals))
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


def sec12_payload(TV, TW, BV, BW, r, Jg, grid3dcap, kfull=SEC12_KFULL, gTV=None, gTW=None, gBV=None, gBW=None, wbases=None):
    """§12 per-sample Hessian eigenvector diagnostics, from per-sample Lanczos eigenpairs. MIRRORS
    server._sec12_payload. TV,BV:(N,K,p) top/bottom eigenVECTORS; TW,BW:(N,K) eigenVALUES; r:(N,) residual;
    Jg:(N,p) per-sample GT-label gradients ∇f_i. Returns {M,do3d,ks,ang,proj}:
      ang.g = the principal-angle (i,j) grids for k=1,2,5,10,kfull (deg); ang.gm = whole-grid means (panel-1
        titles); ang.mn/mx/me = min/max/MEAN over the OFF-DIAGONAL pairs (i≠j; the i=i diagonal angle is 0),
        used by the panel-2/3 evolution curves; ang.ks lists the k per index.
      proj = panels 4/5. {top,bot} = lists over rank (TOP-2 u_{i,1},u_{i,2} ; BOTTOM-2 u_{i,-1},u_{i,-2}). Each rank
        carries TWO quantities: prod:[mean,std]+pvals = the per-sample PRODUCT |⟨J_i,u⟩|·σ·r_i (all samples), and
        cos:[mean,std]+vals = the gradient↔eigvec ALIGNMENT q_i = |⟨J_i,u⟩|/‖J_i‖ (= |cos∠(J_i,u)| ∈ [0,1]; u
        unit-norm; ‖J_i‖=0 dropped). pvals/vals = per-sample arrays for the histograms. MIRRORS index.html sec1213Payload."""
    N, K, p = int(TV.shape[0]), int(TV.shape[1]), int(TV.shape[2])
    dev = TV.device
    offm = ~torch.eye(N, dtype=torch.bool, device=dev)   # off-diagonal (i≠j) mask
    # VALID (real, unit-norm) eigenvectors — on Lanczos breakdown (operator rank < 2K) the surplus eigenpairs are
    # SPURIOUS null-space vectors with norm ≈ 0; every §12 quantity is restricted to span{e:‖e‖≈1} so spurious
    # vectors can't inject phantom 90° angles or collapse the MAX-angle xproj. (No-op when full-rank ⇒ all valid.)
    validT = TV.norm(dim=2) > 0.5; validB = BV.norm(dim=2) > 0.5   # (N,K) each

    # ---- principal angles (gauge-free: svdvals is invariant to eigenvector ±) for k = 1,2,5,10,kfull ----
    #      Averaged over only the REAL principal angles (rij = min(#valid_i,#valid_j) largest cosines).
    def pa(k):
        kk = max(1, min(k, K))
        E = torch.cat([TV[:, :kk, :], BV[:, :kk, :]], dim=1)
        vm = torch.cat([validT[:, :kk], validB[:, :kk]], dim=1)   # (N,2kk) real-eigvec mask for this slice
        Dk = E.shape[1]
        sv = torch.linalg.svdvals(torch.einsum('iap,jbp->ijab', E, E)).clamp(-1.0, 1.0)  # (N,N,2kk) desc by cos
        ang = torch.acos(sv) * (180.0 / math.pi)
        nv = vm.sum(1)
        rij = torch.minimum(nv.view(N, 1), nv.view(1, N))        # (N,N) # of real principal angles
        keep = torch.arange(Dk, device=dev).view(1, 1, Dk) < rij.unsqueeze(2)
        return (ang * keep).sum(2) / keep.sum(2).clamp_min(1)    # (N,N) mean over real principal angles only
    kf = max(1, min(kfull, K))
    ks_ang = [1, 2, 5, 10, kf]
    pas = [pa(k) for k in ks_ang]

    def offstats(g):                                     # min/max/mean over off-diagonal pairs (i≠j)
        s = g[offm]
        return (float(s.min()), float(s.max()), float(s.mean())) if s.numel() else (0.0, 0.0, 0.0)
    om = [offstats(g) for g in pas]
    ang = {"g": [x.reshape(-1).detach().cpu().tolist() for x in pas], "gm": [float(x.mean()) for x in pas],
           "mn": [o[0] for o in om], "mx": [o[1] for o in om], "me": [o[2] for o in om], "ks": ks_ang, "kfull": kf}

    # ---- §12a diagonal alignment: mean_{i≠j} of (1/2k)Σ_a|⟨e_{i,a},e_{j,a}⟩| (same-rank eigvec overlap) for k=1,2,5,15. ----
    def diag_align(k):
        kk = max(1, min(k, K))
        E = torch.cat([TV[:, :kk, :], BV[:, :kk, :]], dim=1)      # (N,2kk,p)
        vm = torch.cat([validT[:, :kk], validB[:, :kk]], dim=1)   # (N,2kk) real-eigvec mask
        d = torch.einsum('iap,jap->ija', E, E).abs()             # (N,N,2kk) |⟨e_{i,a},e_{j,a}⟩|
        both = vm.unsqueeze(1) & vm.unsqueeze(0)                  # (N,N,2kk) same-rank a real in BOTH i and j
        cnt = both.sum(2)
        m = (d * both).sum(2) / cnt.clamp_min(1)                 # (N,N) mean over real same-rank entries only
        sel = offm & (cnt > 0)
        return float(m[sel].mean()) if sel.any() else 0.0
    ks_dal = [1, 2, 5, 15]
    ang["dal"] = [diag_align(k) for k in ks_dal]; ang["dal_ks"] = ks_dal

    # ---- panels 4/5: TOP-2 (u_{i,1},u_{i,2}; largest eigenvalues) and BOTTOM-2 (u_{i,-1},u_{i,-2}; most-negative)
    #      eigvecs of Q_i. TWO per-sample quantities, each mean/std over samples (2 ranks/group → 2 lines per plot):
    #      (a) PRODUCT prod_i = |⟨J_i,u⟩|·σ·r_i (all samples); (b) ALIGNMENT q_i = |⟨J_i,u⟩|/‖J_i‖ = |cos∠| ∈ [0,1]
    #      (‖J_i‖=0 dropped). ----
    ai = torch.arange(N, device=dev)
    nrank = min(2, K)                                            # the top-2 (and bottom-2) eigenvectors → 2 lines/plot
    jnorm = Jg.norm(dim=1)                                       # ‖J_i‖ = per-sample gradient norm
    nz = jnorm > 0                                              # drop samples with ‖J_i‖=0 (can't normalise the alignment)

    def rank_stats(V, W, largest):                             # V/W = (TV,TW) top or (BV,BW) bottom
        idx = W.topk(nrank, dim=1, largest=largest).indices    # (N,nrank): rank 0 = largest (top)/most-negative (bottom), rank 1 = 2nd
        Vn = V.norm(dim=2)                                      # (N,K) eigenvector norms (spurious ≈ 0)
        def ms(x):
            return [float(x.mean()), float(x.std(unbiased=False))] if x.numel() else [0.0, 0.0]
        out = []
        for rk in range(nrank):
            irk = idx[:, rk]
            real = Vn[ai, irk] > 0.5                           # samples whose rk-th eigvec actually EXISTS (unit) vs spurious
            jdot = (Jg * V[ai, irk]).sum(dim=1)                # ⟨J_i,u⟩  (u unit-norm)
            sig = W[ai, irk]                                   # σ (signed eigenvalue)
            prod = ((jdot.abs() * sig) * r)[real]              # PRODUCT |⟨J_i,u⟩|·σ·r_i  (real-eigvec samples)
            q = jdot[nz & real].abs() / jnorm[nz & real]       # ALIGNMENT |⟨J_i,u⟩|/‖J_i‖ = |cos∠(J_i,u)| ∈ [0,1]
            out.append({"prod": ms(prod), "pvals": prod.detach().cpu().tolist(),   # product (+ per-sample for histogram)
                        "cos": ms(q), "vals": q.detach().cpu().tolist()})          # alignment (+ per-sample for histogram)
        return out

    # panels 3/4 (AVERAGED view): each per-sample ∇f_i onto the top-2/bottom-2 eigvecs w of the AVERAGED Hessian (1/N)Σ_i Q_i
    # (shared basis); product |⟨J_i,w⟩|·σ·r_i (σ = global eigenvalue), alignment |⟨J_i,w⟩|/‖J_i‖; mean/std over samples.
    # At N=1 coincides with panels 1/2. MIRRORS server._sec12_payload.
    def gproj_stats(gV, gW):
        def ms(x):
            return [float(x.mean()), float(x.std(unbiased=False))] if x.numel() else [0.0, 0.0]
        o = []
        for rk in range(min(nrank, int(gV.shape[0]))):
            w = gV[rk]; sig = float(gW[rk])
            jdot = Jg @ w
            prod = (jdot.abs() * sig) * r
            q = jdot[nz].abs() / jnorm[nz]
            o.append({"prod": ms(prod), "pvals": prod.detach().cpu().tolist(),
                      "cos": ms(q), "vals": q.detach().cpu().tolist()})
        return o

    out = {"M": N, "do3d": False, "ks": [1, 2, 5], "ang": ang,
           "proj": {"top": rank_stats(TV, TW, True), "bot": rank_stats(BV, BW, False)}}
    if gTV is not None and gBV is not None:
        out["gproj"] = {"top": gproj_stats(gTV, gTW), "bot": gproj_stats(gBV, gBW)}
    if wbases is not None:                                       # panels 5/6/7 — WEIGHTED-AVERAGE Hessian (Σ w_i Q_i)/(Σ w_i)
        out["wproj"] = [{"top": gproj_stats(b["tv"], b["tw"]), "bot": gproj_stats(b["bv"], b["bw"])} for b in wbases]

    # ---- §12b new panels: SVD-principal-direction → nearest-eigenvector CROSS projections (MIN / MAX angle).
    #      Per OFF-DIAGONAL pair (i≠j): SVD of the REAL-subspace overlap E_i^✓E_j^✓ᵀ (✓ = valid unit eigvecs only);
    #      ℓ=0=MIN angle / ℓ=rank-1=MAX (rank=min(#valid_i,#valid_j)); u_i=e_{i,argmax|U[:,ℓ]|},
    #      u_j=e_{j,argmax|V[:,ℓ]|} both unit. |⟨J,u⟩| sign-free, σ,r signed; mean over real-overlap i≠j pairs.
    #      MIRRORS server._sec12_payload. ----
    Ef = torch.cat([TV, BV], dim=1); Wf = torch.cat([TW, BW], dim=1)
    valid = torch.cat([validT, validB], dim=1)                  # (N,2K) real (unit) vs spurious (~0)
    jnc = jnorm.clamp_min(1e-30)
    accmin = [0.0, 0.0, 0.0, 0.0]; accmax = [0.0, 0.0, 0.0, 0.0]; cnt = 0; nis = 0
    vidx = [torch.nonzero(valid[i], as_tuple=False).flatten() for i in range(N)]
    for i in range(N):
        if vidx[i].numel() == 0:
            continue
        i_contrib = False
        for j in range(N):
            if i == j or vidx[j].numel() == 0:
                continue
            Ei = Ef[i][vidx[i]]; Ej = Ef[j][vidx[j]]           # real subspaces
            U, _, Vh = torch.linalg.svd(Ei @ Ej.t())
            rank = min(int(vidx[i].numel()), int(vidx[j].numel()))
            for acc, col in ((accmin, 0), (accmax, rank - 1)):
                a = int(vidx[i][int(U[:, col].abs().argmax())])
                b = int(vidx[j][int(Vh[col, :].abs().argmax())])
                ui = Ef[i][a]; uj = Ef[j][b]
                si = float(Wf[i][a]); sj = float(Wf[j][b]); ri = float(r[i]); rj = float(r[j])
                ni = float(jnc[i]); nj = float(jnc[j])
                Jiuj = float((Jg[i] * uj).sum().abs()); Jjui = float((Jg[j] * ui).sum().abs())
                acc[0] += Jiuj * si * ri                        # q1,q2: SUMMED over j (meaned over i below)
                acc[1] += Jiuj * Jjui * si * sj * ri * rj
                acc[2] += Jiuj / ni                            # q3,q4: meaned over all i≠j pairs
                acc[3] += Jiuj * Jjui / (ni * nj)
            cnt += 1; i_contrib = True
        if i_contrib:
            nis += 1
    dvp = cnt if cnt else 1; dvi = nis if nis else 1            # q1,q2 ÷ #i (mean_i Σ_j) ; q3,q4 ÷ #pairs (mean_{i≠j})
    out["xproj"] = {"min": [accmin[0] / dvi, accmin[1] / dvi, accmin[2] / dvp, accmin[3] / dvp],
                    "max": [accmax[0] / dvi, accmax[1] / dvi, accmax[2] / dvp, accmax[3] / dvp]}
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
        P = (1.0 + pr.view(N, 1) * (lr / N) * Wp) * Jp              # (1+r_{t-1}·(η/N)·σ_{t-1})·J'_{t-1}  (η/N = the 1/N-normalized GD step; i- & k-side)
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


def sec14_payload(prev, cur, lr, grid3dcap, rhist):
    """§14 — per-triplet (i,j,k) decomposition of Tr(ΔNTK)=Σ_{ℓ,p} a·b·c·d·e (a=u_{i,ℓ}·u_{j,p}, b=σ_{i,ℓ}J_i·u_{i,ℓ},
    c=1+ησ_{j,p}r_j, d=u_{j,p}·J_k, e=r_k). TIME-INDEXING (t+1=current iterate): u,σ,r_j(c),J_k(d) at the PREVIOUS
    eig-tick t; only J_i(b) and r_k(e) at t+1. So §14 needs both ticks (prev,cur={TV,TW,BV,BW,r,Jg}) and skips its
    first eig-tick (like §13); eigvecs are gauged by the PREVIOUS gradient. Rank-2y; the (2y)² terms sorted
    signed-descending per cell. Panels per y∈{1,2,5}: agg=[max,min,Σtop-y,Σbot-y,Σall]; max/min=[a,b·d,a·b·d,a·b·d·e,
    a·e] at the argmax/argmin. Panel 10 UNCHANGED — current-tick σ_{j,p}: ∏_{z=1}^{S}(1+ησ_{j,p}r_{j,t-S+z}) (S=y)
    over the residual history, first/last × y. DENSE here; parity-matches server._sec14_payload / index.html sec14Payload."""
    TVp, TWp, BVp, BWp, rp, Jgp = prev["TV"], prev["TW"], prev["BV"], prev["BW"], prev["r"], prev["Jg"]
    TVc, TWc, BVc, BWc, rc, Jgc = cur["TV"], cur["TW"], cur["BV"], cur["BW"], cur["r"], cur["Jg"]
    N, K, p = int(TVp.shape[0]), int(TVp.shape[1]), int(TVp.shape[2])
    out = {"M": N, "do3d": False}
    if N > grid3dcap:
        return out
    out["do3d"] = True
    dev = TVp.device
    dg = torch.arange(N, device=dev)

    def pack14(T):
        f = T.reshape(-1); pos = f[f > 0]; neg = f[f < 0]
        return {"v": f.detach().cpu().tolist(), "idx": None, "d": T[dg, dg, dg].detach().cpu().tolist(),
                "mn": float(f.mean()), "mp": float(pos.mean()) if pos.numel() else 0.0,
                "mng": float(neg.mean()) if neg.numel() else 0.0}

    def build(TVx, TWx, BVx, BWx, Jgg, Jgb, rcc, Jgd, ree, y):
        """rank-2y gauged factors with per-factor TIME sources: eigvecs/σ from (TVx..); gauge grad Jgg; b's J_i=Jgb;
        c's r_j=rcc; d's J_k=Jgd; e's r_k=ree."""
        kk = max(1, min(y, K)); D = 2 * kk
        E = torch.cat([TVx[:, :kk, :], BVx[:, :kk, :]], dim=1)
        W = torch.cat([TWx[:, :kk], BWx[:, :kk]], dim=1)
        gE = torch.einsum('ip,idp->id', Jgg, E)
        E = E * torch.where(gE >= 0, torch.ones_like(gE), -torch.ones_like(gE)).unsqueeze(2)   # gradient gauge (prev grad)
        vE = E.norm(dim=2) > 0.5                       # (N,D) REAL (unit) vs spurious (0) eigvec mask
        a = torch.einsum('ilp,jmp->ijlm', E, E)
        b = W * torch.einsum('ip,idp->id', Jgb, E)   # J_i ← Jgb
        c = 1.0 + (lr / N) * W * rcc.view(N, 1)       # r_j ← rcc ; η/N = 1/N-normalized GD step (matches §13)
        d = torch.einsum('jmp,kp->jmk', E, Jgd)       # J_k ← Jgd
        e = ree                                       # r_k ← ree
        T = (a.unsqueeze(2) * b.view(N, 1, 1, D, 1) * c.view(1, N, 1, 1, D)
             * d.permute(0, 2, 1).reshape(1, N, N, 1, D) * e.view(1, 1, N, 1, 1))
        return a, b, d, e, W, T, D, kk, vE

    def argsel(Tf, D):
        ar = torch.arange(D * D, device=dev).view(1, 1, 1, D * D); big = torch.full((), D * D, device=dev, dtype=ar.dtype)
        imax = torch.where(Tf == Tf.amax(3, keepdim=True), ar, big).amin(3).clamp_(max=D * D - 1)   # clamp: NaN cell ⇒ no match ⇒ big
        imin = torch.where(Tf == Tf.amin(3, keepdim=True), ar, big).amin(3).clamp_(max=D * D - 1)
        return imax, imin

    inf = float("inf")
    def modemask(vE, D):                              # (N,N,N,D²): mode (ℓ,p) REAL iff u_{i,ℓ} & u_{j,p} both real (∀k)
        return (vE.view(N, 1, D, 1) & vE.view(1, N, 1, D)).reshape(N, N, D * D).unsqueeze(2).expand(N, N, N, D * D)
    def finite(x):
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x))

    p14 = {}; sig_sel = {"first": {}, "last": {}}
    for y in SEC14_YS:
        # panels 1-9: u,σ,r_j,J_k at t (prev); J_i(b)=cur, r_k(e)=cur; gauge = prev gradient
        a, b, d, e, W, T, D, kk, vE = build(TVp, TWp, BVp, BWp, Jgp, Jgc, rp, Jgp, rc, y)
        Tf = T.reshape(N, N, N, D * D)
        vm = modemask(vE, D)                          # SPURIOUS modes (zero-norm eigvec) excluded from sort/select
        Tmax = Tf.masked_fill(~vm, -inf); Tmin = Tf.masked_fill(~vm, inf)
        Tsx, _ = torch.sort(Tmax, dim=3, descending=True); Tsn, _ = torch.sort(Tmin, dim=3, descending=False)
        agg = [finite(Tsx[..., 0]), finite(Tsn[..., 0]),
               finite(Tsx[..., :kk]).sum(3), finite(Tsn[..., :kk]).sum(3), Tf.sum(3)]
        imax = argsel(Tmax, D)[0]; imin = argsel(Tmin, D)[1]

        def subprods(idx):
            lst = idx // D; mst = idx % D
            a_sel = a.reshape(N, N, D * D).unsqueeze(2).expand(N, N, N, D * D).gather(3, idx.unsqueeze(3)).squeeze(3)
            b_sel = b.view(N, 1, 1, D).expand(N, N, N, D).gather(3, lst.unsqueeze(3)).squeeze(3)
            d_sel = d.permute(0, 2, 1).reshape(1, N, N, D).expand(N, N, N, D).gather(3, mst.unsqueeze(3)).squeeze(3)
            e_sel = e.view(1, 1, N).expand(N, N, N)
            return [a_sel, b_sel * d_sel, a_sel * b_sel * d_sel, a_sel * b_sel * d_sel * e_sel, a_sel * e_sel]
        sub_max = subprods(imax); sub_min = subprods(imin)
        p14[str(y)] = {"agg": [pack14(x) for x in agg], "max": [pack14(x) for x in sub_max],
                       "min": [pack14(x) for x in sub_min]}

        # panel 10 (eq-③): all-current term → its argmax/argmin REAL mode + current σ_{j,p}
        _, _, _, _, Wc, Tc, Dc, _, vEc = build(TVc, TWc, BVc, BWc, Jgc, Jgc, rc, Jgc, rc, y)
        Tcf = Tc.reshape(N, N, N, Dc * Dc); vmc = modemask(vEc, Dc)
        imax_c = argsel(Tcf.masked_fill(~vmc, -inf), Dc)[0]; imin_c = argsel(Tcf.masked_fill(~vmc, inf), Dc)[1]
        Wjc = Wc.view(1, N, 1, Dc).expand(N, N, N, Dc)
        sig_sel["first"][str(y)] = Wjc.gather(3, (imax_c % Dc).unsqueeze(3)).squeeze(3)
        sig_sel["last"][str(y)] = Wjc.gather(3, (imin_c % Dc).unsqueeze(3)).squeeze(3)
    out["p14"] = p14

    ratios = []
    for sel in ("first", "last"):
        for y in SEC14_YS:
            S = y
            if len(rhist) < S:
                ratios.append(None); continue
            sig = sig_sel[sel][str(y)]
            prod = torch.ones(N, N, N, device=dev, dtype=TVp.dtype)
            for rt in rhist[-S:]:
                prod = prod * (1.0 + (lr / N) * sig * rt.view(1, N, 1))   # η/N (1/N-normalized GD step; matches §13)
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


# ───────────────────────── §15 — 2nd-difference decomposition of ‖J‖²_F and σ₁ ─────────────────────────
# Over GD with step Δθ = (η/N)·Σᵢrᵢ∇fᵢ, the discrete 2nd difference of ‖J‖²_F (= tr NTK) and of σ₁ (top NTK
# eigenvalue) is decomposed into theory terms I/II/III (resp. IV/V/VI) plus the matrices A (resp. B). All
# per-sample function-Hessian actions Q_{s,k}·v come from the callables hvp1 (at t−1) and hvp2 (at t−2);
# Jt/Jtm1/Jtm2 are (N,p) per-sample GT-gradients (row k = ∇f_{s,k}), rtm1/rtm2 are (N,). c = η/N.
#   I  = c²·g_{t−1}ᵀ S_{t−1} g_{t−1},  S = Σ_k Q_{t−1,k}²,  g_s = Σᵢ r_{s,i}∇f_{s,i}
#   II = c²·Σ_k ∇f_{t,k}ᵀ Q_{t−1,k}·Q̃·g_{t−2},  Q̃ = Σⱼ r_{t−1,j} Q_{t−2,j}
#   III= c²·Σ_k ∇f_{t,k}ᵀ Q_{t−1,k}·(J_{t−1} J_{t−2}ᵀ J_{t−2} r_{t−2})
#   A  = c²·(J_{t−1}ᵀ S_{t−1} J_{t−1})(I − c·J_{t−2}ᵀJ_{t−2})       (N×N, real eigenvalues: PSD·symmetric)
#   IV/V/VI/B: the σ₁ analogs — S→(Q̄ᵘ)² with Q̄ᵘ=Σ_k u_{t,1,k}Q_{t−1,k}, and the trace-tie → u_{t,1} projection.
_DREG = 1e-12   # the divergence divides by the 2nd difference D² (resp Dσ²), which can hit 0 at an inflection of ‖J‖²/σ₁.
def _divreg(num, den, scale):   # regularize the DENOMINATOR by a tiny perturbation ~roundoff·scale so it is defined at D²=0
    return num / (den + math.copysign(_DREG * scale, den)) * 100.0   # absorbs float cancellation; value stays UNBOUNDED (no clipping)
def sec15_stats(hvp1, hvp2, Jt, Jtm1, Jtm2, rtm1, rtm2, lr, N):
    dt, dev = Jt.dtype, Jt.device
    c = lr / N
    c2 = c * c

    def estats(M):                                            # real eigenvalues of N×N M (A,B are PSD·sym ⇒ real)
        e = torch.linalg.eigvals(M).real
        es = torch.sort(e, descending=True).values
        k = min(3, es.numel())
        top = es[:k].tolist()
        bot = torch.sort(e).values[:k].tolist()               # 3 smallest, ascending
        st = [float(e.min()), float(e.max()), float(e.mean()),
              float(torch.quantile(e, 0.25)), float(torch.quantile(e, 0.75))]
        return top, bot, st, es.tolist()                      # full spectrum (desc) for the SLQ density (plot 5)

    g1 = Jtm1.t() @ rtm1                                      # g_{t−1}  (p,)
    g2 = Jtm2.t() @ rtm2                                      # g_{t−2}  (p,)
    Ktm2 = Jtm2 @ Jtm2.t()                                    # NTK at t−2  (N,N)
    Imat = torch.eye(N, dtype=dt, device=dev)

    H1g1 = hvp1(g1)                                           # (N,p) row k = Q_{t−1,k} g_{t−1}
    I = c2 * float((H1g1 * H1g1).sum())                       # = c²·Σ_k‖Q_{t−1,k}g_{t−1}‖² = c²·g₁ᵀSg₁

    W = torch.stack([hvp1(Jtm1[i]) for i in range(N)])        # (N_i,N_k,p)  W[i,k]=Q_{t−1,k}∇f_{t−1,i}
    JSJ = torch.einsum('ikp,lkp->il', W, W)                   # J_{t−1}ᵀ S_{t−1} J_{t−1}  (N,N)
    # (*) term so that I+II = r_{t−1}ᵀ A r_{t−2}:  A_part2[j,i] = c²·Σ_k ∇f_{t,k}ᵀ Q_{t−1,k} Q_{t−2,j} ∇f_{t−2,i}
    #   = c²·⟨ψ, Q_{t−2,j}∇f_{t−2,i}⟩  with  ψ = Σ_k Q_{t−1,k}∇f_{t,k};  M2[i,j] = Q_{t−2,j}∇f_{t−2,i}
    psi = torch.stack([hvp1(Jt[k])[k] for k in range(N)]).sum(0)   # Σ_k Q_{t−1,k}∇f_{t,k}  (p,)
    M2 = torch.stack([hvp2(Jtm2[i]) for i in range(N)])            # (i,j,p)  M2[i,j]=Q_{t−2,j}∇f_{t−2,i}
    A = c2 * (JSJ @ (Imat - c * Ktm2)) + c2 * torch.einsum('a,ija->ji', psi, M2)   # general (non-symmetric) ⇒ real-part eigenvalues
    Atop, Abot, Astat, Aeig = estats(A)

    a = rtm1 @ hvp2(g2)                                       # Q̃ g_{t−2}  (p,)
    H1a = hvp1(a)
    II = c2 * float((Jt * H1a).sum())                         # Σ_k ∇f_{t,k}·(Q_{t−1,k} a)

    w = Ktm2 @ rtm2                                           # J_{t−2}ᵀJ_{t−2} r_{t−2}  (N,)
    z = Jtm1.t() @ w                                          # J_{t−1} w  (p,)
    H1z = hvp1(z)
    III = c2 * float((Jt * H1z).sum())

    nJt = float((Jt * Jt).sum()); nJtm1 = float((Jtm1 * Jtm1).sum()); nJtm2 = float((Jtm2 * Jtm2).sum())
    D2 = nJt + nJtm2 - 2.0 * nJtm1
    divP = _divreg(-2.0 * (I + II - III) + D2, D2, nJt + nJtm1 + nJtm2)  # 2×: terms are ½∂²‖J‖²; D² is the full discrete ∂² ⇒ predict D²≈2(I+II−III); →0 when theory holds

    # ── panel 2 (σ₁): top NTK eigenpair at t,t−1,t−2 + the u₁-projected terms ──
    Kt = Jt @ Jt.t()
    evt = torch.linalg.eigh(Kt)
    sig_t = float(evt.eigenvalues[-1])
    u1 = evt.eigenvectors[:, -1]                              # top NTK eigenvector at t  (N,)
    sig_tm1 = float(torch.linalg.eigvalsh(Jtm1 @ Jtm1.t())[-1])
    sig_tm2 = float(torch.linalg.eigvalsh(Ktm2)[-1])

    Qu_g1 = u1 @ H1g1                                         # Q̄ᵘ g_{t−1}  (p,)
    IV = c2 * float((Qu_g1 * Qu_g1).sum())
    b = u1 @ Jt                                               # J_t u_{t,1}  (p,)
    V = c2 * float((b * (u1 @ H1a)).sum())                    # bᵀ Q̄ᵘ (Q̃ g_{t−2})
    VI = c2 * float((b * (u1 @ H1z)).sum())                   # bᵀ Q̄ᵘ z
    P = torch.einsum('k,ikp->ip', u1, W)                      # Q̄ᵘ ∇f_{t−1,i}  (N,p)
    Bm = c2 * ((P @ P.t()) @ (Imat - c * Ktm2))               # J_{t−1}ᵀ(Q̄ᵘ)²J_{t−1}·(I−cK_{t−2})
    # (**) term so that IV+V = r_{t−1}ᵀ B r_{t−2}:  B_part2[j,i] = c²·⟨Q̄ᵘ b, Q_{t−2,j}∇f_{t−2,i}⟩,  b=Σ_k u₁ₖ∇f_{t,k}
    Bm = Bm + c2 * torch.einsum('a,ija->ji', u1 @ hvp1(b), M2)   # Q̄ᵘb = u1·{Q_{t−1,a}b}; reuses M2
    Btop, Bbot, Bstat, Beig = estats(Bm)
    Dsig2 = sig_t + sig_tm2 - 2.0 * sig_tm1
    divS = _divreg(-2.0 * (IV + V - VI) + Dsig2, Dsig2, sig_t + sig_tm1 + sig_tm2)  # 2×: terms are ½∂²σ₁; Dσ² is the full discrete ∂² ⇒ predict Dσ²≈2(IV+V−VI); →0 when theory holds

    rec = {"I": I, "II": II, "III": III, "divP": divP, "Atop": Atop, "Abot": Abot, "Astat": Astat,
           "IV": IV, "V": V, "VI": VI, "divS": divS, "Btop": Btop, "Bbot": Bbot, "Bstat": Bstat,
           "D2": D2, "Dsig2": Dsig2, "Aeig": Aeig, "Beig": Beig,
           "Js": nJt + nJtm1 + nJtm2, "Ss": sig_t + sig_tm1 + sig_tm2}   # Aeig/Beig: full real spectra → SLQ density (plot 5)
    return rec, A, Bm, u1   # also expose the matrices A,B and the top NTK eigvec u₁ for panels 3/4 (Q̇≠0)


def _eig_stats(M):
    """Real-part eigenvalue summary of a general N×N matrix: (top-3 desc, bottom-3 asc, [min,max,mean,q25,q75], full desc)."""
    e = torch.linalg.eigvals(M).real
    es = torch.sort(e, descending=True).values
    k = min(3, es.numel())
    return (es[:k].tolist(), torch.sort(e).values[:k].tolist(),
            [float(e.min()), float(e.max()), float(e.mean()), float(torch.quantile(e, 0.25)), float(torch.quantile(e, 0.75))],
            es.tolist())


def sec15_panel34(hvp_at, th_tm2, Jt, Jtm1, Jtm2, rtm1, rtm2, u1, A, B, rec, lr, N, eps=1e-3):
    """Panels 3/4 (Q̇≠0): the third-derivative terms VII, VIII and the matrices A'=A+(VII), B'=B+(VIII).
    `hvp_at(theta, v)` returns the per-sample function-Hessian action {Q_{s,k}(theta)·v}_k (N,p) at an ARBITRARY
    theta — used to take a directional derivative of the Hessian (finite difference of the FD-HVP, the 3rd derivative
    T_{t-2,k}).  VII = c²·Σ_k ∇f_{t,k}ᵀ T_{t-2,k}[·,g_{t-1},g_{t-2}] ;  VIII = the u₁-projected analogue.
      A'[j,i] = A[j,i] + c²·Σ_k ∇f_{t,k}ᵀ T_{t-2,k}[·,∇f_{t-1,j},∇f_{t-2,i}]  (and B' the u₁-projected analogue).
    Costs 2N²+2 HVP evaluations (the A'/B' matrices dominate)."""
    c = lr / N; c2 = c * c
    g1 = Jtm1.t() @ rtm1; g2 = Jtm2.t() @ rtm2
    b = u1 @ Jt                                              # Σ_k u₁ₖ ∇f_{t,k}  (p,)
    # ── scalars VII, VIII: directional derivative of {Q_k·g₁} along g₂ (share Hp/Hm) ──
    dQg1 = (hvp_at(th_tm2 + eps * g2, g1) - hvp_at(th_tm2 - eps * g2, g1)) / (2.0 * eps)   # {T_{t-2,k}[·,g₁,g₂]}_k
    VII = c2 * float((Jt * dQg1).sum())
    VIII = c2 * float((b * (u1 @ dQg1)).sum())
    # ── matrices A'−A, B'−B: T_{t-2,k}[·, ∇f_{t-1,j}, ∇f_{t-2,i}] over all (i,j) ──
    dA = torch.zeros(N, N, dtype=Jt.dtype, device=Jt.device)
    dB = torch.zeros(N, N, dtype=Jt.dtype, device=Jt.device)
    for i in range(N):
        thp = th_tm2 + eps * Jtm2[i]; thm = th_tm2 - eps * Jtm2[i]
        for j in range(N):
            d = (hvp_at(thp, Jtm1[j]) - hvp_at(thm, Jtm1[j])) / (2.0 * eps)   # {T_{t-2,k}[·,∇f_{t-1,j},∇f_{t-2,i}]}_k
            dA[j, i] = (Jt * d).sum()
            dB[j, i] = (b * (u1 @ d)).sum()
    Ap = A + c2 * dA; Bp = B + c2 * dB
    Aptop, Apbot, Apstat, Apeig = _eig_stats(Ap)
    Bptop, Bpbot, Bpstat, Bpeig = _eig_stats(Bp)
    D2 = rec["D2"]; Dsig2 = rec["Dsig2"]
    sJ = rec["I"] + rec["II"] - rec["III"] + VII             # full ½∂²‖J‖² with the Q̇ term
    sS = rec["IV"] + rec["V"] - rec["VI"] + VIII             # full ½∂²σ₁ with the Q̇ term
    divP3 = _divreg(-2.0 * sJ + D2, D2, rec["Js"])
    divS4 = _divreg(-2.0 * sS + Dsig2, Dsig2, rec["Ss"])
    return {"VII": VII, "VIII": VIII, "divP3": divP3, "divS4": divS4,
            "Aptop": Aptop, "Apbot": Apbot, "Apstat": Apstat, "Apeig": Apeig,
            "Bptop": Bptop, "Bpbot": Bpbot, "Bpstat": Bpstat, "Bpeig": Bpeig}
