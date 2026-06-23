#!/usr/bin/env python3
"""GPU backend for the EoS / Progressive-Sharpening widget.

Serves `index_server.html` and streams the per-step training diagnostics — loss,
sharpness, top/bottom eigenvalues of H / loss-Hessian / G / S, J-projections, SLQ
densities, eigenspace rotation — computed in **PyTorch on a GPU**, over Server-Sent
Events (SSE). The browser keeps doing only the (fast) Plotly drawing.

Why this exists: `index.html` is zero-backend and runs every HVP / Lanczos / SLQ in
single-threaded browser JS on the CPU, which is the entire bottleneck. Here the math
runs natively on the idle GPUs of this machine; only tiny JSON messages cross the wire.

Pure standard-library HTTP (no flask/fastapi/websockets needed) + torch.

Numerics: data + initialization use the exact same `mulberry32` RNG as index.html /
_preview.py, so the GD trajectory is identical to the browser for a given seed. The
Lanczos / SLQ probe vectors use torch's RNG (fast); extreme eigenvalues converge to
the same values regardless of start vector, so the plots match the browser closely.

Run:
    python3 server.py                      # auto-picks the freest GPU, fp32
    python3 server.py --device cuda:1      # pin a GPU
    python3 server.py --dtype float64      # match the browser's double precision exactly
    python3 server.py --host 0.0.0.0 --port 8000

Then (from your laptop) tunnel one port and open it:
    ssh -N -L 8000:localhost:8000 <this-box>
    open http://localhost:8000/
"""
import argparse
import collections
import json
import math
import mimetypes
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import threading

import torch
import torch.nn.functional as F

DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cpu")   # default/fallback device (= DEVICE_POOL[0]); set in main()
DEVICE_POOL = []               # every device a run may be assigned to (one GPU each); built in main()
DTYPE = torch.float64          # set in main()
EPS = 1e-3                     # finite-difference step (matches index.html / _preview.py)
G3D_MAXPTS = 10000             # §11 render budget: emit only ≤this many points per grid above this size
                               #   (the tensors are heavy-tailed — ~90% near-zero — keeps M up to ~500 renderable)
PMAX = 10_000_000              # hard cap on parameter count p (matrix-free, so memory is O(p))


def _g3d_pack(flat, K=G3D_MAXPTS):
    """≤K points spanning the value range of a heavy-tailed flat tensor → (values, indices | None).
    Half the budget is the top-|value| points (structure); half is a random sample of the near-zero bulk —
    so the rendered colours show a fine white→saturated gradient instead of only the (uniform) saturated tail."""
    n = int(flat.numel())
    if n <= K:
        return flat.detach().cpu().tolist(), None
    ktop = K // 2
    top = torch.topk(flat.abs(), ktop).indices
    rnd = torch.randint(0, n, (K - ktop,), device=flat.device)   # random bulk (with-replacement; dups are harmless)
    idx = torch.cat([top, rnd])
    return flat[idx].detach().cpu().tolist(), idx.to(torch.int64).cpu().tolist()


# ============================ §12 per-sample Hessian eigenvector cross-similarity ============================
# For each sample i, Q_i = ∇²f_i (the ground-truth-label output's Hessian). Its top-K and bottom-K eigenpairs
# are extracted MATRIX-FREE by Lanczos on the per-sample HVP v ↦ Q_i·v (NO p×p matrix is ever formed — scales
# to any p, like §1–§3). _sec12_payload then compares those eigenvectors across sample pairs (i,j). Pure &
# tensor-only so it is unit-testable; MIRRORS eos_lab.linalg.sec12_payload / index.html sec12Payload.
SEC12_KFULL = 15   # rank of the "full-space" principal-angle plot (top-15 ⊕ bottom-15) — a fixed cap so it,
                   # too, needs only Lanczos extremes rather than the full spectrum.


def _sec12_payload(TV, TW, BV, BW, r, Jg, grid3dcap, kfull=SEC12_KFULL, gTV=None, gTW=None, gBV=None, gBW=None, wbases=None):
    """§12 per-sample Hessian eigenvector diagnostics, from per-sample Lanczos eigenpairs.
    TV,BV:(N,K,p) top/bottom eigenVECTORS; TW,BW:(N,K) eigenVALUES; r:(N,) residual; Jg:(N,p) per-sample
    GT-label gradients ∇f_i. Returns {M,do3d,ks,ang,proj}:
      ang.g = the principal-angle (i,j) grids for k=1,2,5,10,kfull (deg); ang.gm = whole-grid means (panel-1
        titles); ang.mn/mx/me = min/max/MEAN over the OFF-DIAGONAL pairs (i≠j; the i=i diagonal angle is 0),
        used by the panel-2/3 evolution curves; ang.ks lists the k per index.
      proj = panels 4/5. For the TOP-2 eigvecs u_{i,1},u_{i,2} (largest eigenvalues → proj.top, a list of 2 ranks)
        and the BOTTOM-2 u_{i,-1},u_{i,-2} (most-negative → proj.bot) of Q_i, each rank carries TWO quantities:
          • prod:[mean,std], pvals — the per-sample PRODUCT prod_i = |⟨J_i,u⟩|·σ·r_i (all samples; row-1 cols 1-2),
          • cos:[mean,std], vals   — the per-sample gradient↔eigvec ALIGNMENT q_i = |⟨J_i,u⟩|/‖J_i‖ (= |cos∠(J_i,u)|
            ∈ [0,1]; u unit-norm; ‖J_i‖=0 dropped; row-1 cols 3-4).
        pvals/vals = the per-sample arrays for the row-2 histograms. MIRRORS eos_lab.linalg.sec12_payload /
        index.html sec1213Payload."""
    import math as _m
    N, K, p = int(TV.shape[0]), int(TV.shape[1]), int(TV.shape[2])
    dev = TV.device
    offm = ~torch.eye(N, dtype=torch.bool, device=dev)   # off-diagonal (i≠j) mask
    # VALID (real, unit-norm) eigenvectors. On Lanczos breakdown (per-sample operator rank < 2K — common for
    # shallow/narrow ∇²f_i) the surplus requested eigenpairs are SPURIOUS null-space vectors returned with norm ≈ 0
    # (see batched_lanczos_extreme's clamp). Including them poisons §12: they add bogus 90° principal angles and,
    # worst, the MAX-angle xproj direction SELECTS a zero-norm eigvec ⇒ ⟨J,u⟩≈0 collapsed q1–q4 to 0. Every §12
    # quantity below is therefore restricted to the real subspace span{e : ‖e‖≈1}. (No-op when full-rank ⇒ all valid.)
    validT = TV.norm(dim=2) > 0.5; validB = BV.norm(dim=2) > 0.5   # (N,K) each

    # ---- principal angles (gauge-free: svdvals is invariant to eigenvector ±) for k = 1,2,5,10,kfull ----
    #      Averaged over only the REAL principal angles (the rij = min(#valid_i,#valid_j) largest cosines); the
    #      surplus svdvals≈0 from spurious eigvecs would otherwise inject phantom 90° angles into the mean/max.
    def pa(k):
        kk = max(1, min(k, K))
        E = torch.cat([TV[:, :kk, :], BV[:, :kk, :]], dim=1)
        vm = torch.cat([validT[:, :kk], validB[:, :kk]], dim=1)   # (N,2kk) real-eigvec mask for this slice
        Dk = E.shape[1]
        sv = torch.linalg.svdvals(torch.einsum('iap,jbp->ijab', E, E)).clamp(-1.0, 1.0)  # (N,N,2kk) desc by cos
        ang = torch.acos(sv) * (180.0 / _m.pi)
        nv = vm.sum(1)                                           # (N,) valid count in this slice
        rij = torch.minimum(nv.view(N, 1), nv.view(1, N))        # (N,N) # of real principal angles
        keep = torch.arange(Dk, device=dev).view(1, 1, Dk) < rij.unsqueeze(2)   # first rij (largest cos)
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

    # ---- §12a diagonal alignment: mean_{i≠j} of (1/2k)Σ_a|⟨e_{i,a},e_{j,a}⟩| (same-rank eigvec overlap) for k=1,2,5,15.
    #      |diag(E_iE_jᵀ)| → mean over the 2k entries → mean over off-diagonal pairs. ∈[0,1]; 1 = same-rank eigvecs aligned. ----
    def diag_align(k):
        kk = max(1, min(k, K))
        E = torch.cat([TV[:, :kk, :], BV[:, :kk, :]], dim=1)      # (N,2kk,p)
        vm = torch.cat([validT[:, :kk], validB[:, :kk]], dim=1)   # (N,2kk) real-eigvec mask
        d = torch.einsum('iap,jap->ija', E, E).abs()             # (N,N,2kk) |⟨e_{i,a},e_{j,a}⟩|
        both = vm.unsqueeze(1) & vm.unsqueeze(0)                  # (N,N,2kk) same-rank a real in BOTH i and j
        cnt = both.sum(2)                                        # (N,N) # of real same-rank pairs
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

    # ---- panels 3/4 (AVERAGED view): project each per-sample ∇f_i onto the TOP-2 / BOTTOM-2 eigenvectors w of the
    #      AVERAGED Hessian (1/N)Σ_i Q_i (a single shared basis, ordered top=desc / bottom=most-negative). Same two
    #      quantities as panels 1/2 — product |⟨J_i,w⟩|·σ·r_i (σ = the eigenvalue of (1/N)Σ_i Q_i) and alignment
    #      |⟨J_i,w⟩|/‖J_i‖ — mean/std over samples. At N=1 this coincides with panels 1/2 ((1/N)Σ_i Q_i = Q_1). ----
    def gproj_stats(gV, gW):
        def ms(x):
            return [float(x.mean()), float(x.std(unbiased=False))] if x.numel() else [0.0, 0.0]
        o = []
        for rk in range(min(nrank, int(gV.shape[0]))):
            w = gV[rk]; sig = float(gW[rk])                    # global eigvec + its (global) eigenvalue
            jdot = Jg @ w                                      # (N,) ⟨J_i, w⟩ for every sample
            prod = (jdot.abs() * sig) * r                      # |⟨J_i,w⟩|·σ·r_i
            q = jdot[nz].abs() / jnorm[nz]                     # |⟨J_i,w⟩|/‖J_i‖ ∈ [0,1]
            o.append({"prod": ms(prod), "pvals": prod.detach().cpu().tolist(),
                      "cos": ms(q), "vals": q.detach().cpu().tolist()})
        return o

    out = {"M": N, "do3d": False, "ks": [1, 2, 5], "ang": ang,
           "proj": {"top": rank_stats(TV, TW, True), "bot": rank_stats(BV, BW, False)}}
    if gTV is not None and gBV is not None:                     # panels 3/4 — averaged-Hessian view (1/N)Σ_i Q_i
        out["gproj"] = {"top": gproj_stats(gTV, gTW), "bot": gproj_stats(gBV, gBW)}
    if wbases is not None:                                       # panels 5/6/7 — WEIGHTED-AVERAGE Hessian (Σ w_i Q_i)/(Σ w_i)
        out["wproj"] = [{"top": gproj_stats(b["tv"], b["tw"]), "bot": gproj_stats(b["bv"], b["bw"])} for b in wbases]

    # ---- §12b panels 5/6: SVD-principal-direction → nearest-eigenvector CROSS projections (MIN / MAX angle) ----
    #      For every OFF-DIAGONAL pair (i≠j): SVD of the REAL-subspace overlap E_i^✓E_j^✓ᵀ (✓ = the valid, unit-norm
    #      eigvecs only — spurious null-space rows are dropped so the MAX-angle direction is a genuine least-aligned
    #      REAL principal direction, not a zero-norm vector). For the col-ℓ principal direction (ℓ=0 = MIN angle /
    #      ℓ=rank-1 = MAX angle, rank=min(#valid_i,#valid_j)), pick u_i=e_{i,a*} (a*=argmax_a|U[a,ℓ]|) and
    #      u_j=e_{j,b*} (b*=argmax_b|V[b,ℓ]|) — both GUARANTEED unit; σ_i,σ_j = their eigenvalues. Quantities (|⟨J,u⟩|
    #      sign-free; σ,r signed): q1=|⟨J_i,u_j⟩|σ_i r_i ; q2=|⟨J_i,u_j⟩||⟨J_j,u_i⟩|σ_iσ_j r_i r_j ; q3=|⟨J_i,u_j⟩|/‖J_i‖ ;
    #      q4=|⟨J_i,u_j⟩||⟨J_j,u_i⟩|/(‖J_i‖‖J_j‖). q1,q2 are aggregated as MEAN over i of the SUM over j≠i (per-i total
    #      curvature⟶residual coupling); q3,q4 as the MEAN over all i≠j pairs. ----
    Ef = torch.cat([TV, BV], dim=1)                              # (N,2K,p) full top-K ⊕ bottom-K eigenvectors
    Wf = torch.cat([TW, BW], dim=1)                              # (N,2K) signed eigenvalues
    valid = torch.cat([validT, validB], dim=1)                  # (N,2K) real (unit) vs spurious (~0) eigvecs
    jnc = jnorm.clamp_min(1e-30)                                # ‖J_i‖ (guard /0)
    accmin = [0.0, 0.0, 0.0, 0.0]; accmax = [0.0, 0.0, 0.0, 0.0]; cnt = 0; nis = 0
    vidx = [torch.nonzero(valid[i], as_tuple=False).flatten() for i in range(N)]   # real-eigvec indices per sample
    for i in range(N):
        if vidx[i].numel() == 0:                                # sample i has NO real eigvecs (operator ≈ 0) → no overlap
            continue
        i_contrib = False
        for j in range(N):
            if i == j or vidx[j].numel() == 0:
                continue
            Ei = Ef[i][vidx[i]]; Ej = Ef[j][vidx[j]]           # real subspaces (ri,p),(rj,p)
            U, _, Vh = torch.linalg.svd(Ei @ Ej.t())           # principal directions of the REAL subspaces
            rank = min(int(vidx[i].numel()), int(vidx[j].numel()))
            for acc, col in ((accmin, 0), (accmax, rank - 1)):  # ℓ=0 min angle ; ℓ=rank-1 max angle (real overlap)
                a = int(vidx[i][int(U[:, col].abs().argmax())])    # nearest REAL eigvec of i
                b = int(vidx[j][int(Vh[col, :].abs().argmax())])   # nearest REAL eigvec of j
                ui = Ef[i][a]; uj = Ef[j][b]                    # both unit by construction
                si = float(Wf[i][a]); sj = float(Wf[j][b]); ri = float(r[i]); rj = float(r[j])
                ni = float(jnc[i]); nj = float(jnc[j])
                Jiuj = float((Jg[i] * uj).sum().abs()); Jjui = float((Jg[j] * ui).sum().abs())
                acc[0] += Jiuj * si * ri                        # q1,q2: SUMMED over j (then meaned over i below)
                acc[1] += Jiuj * Jjui * si * sj * ri * rj
                acc[2] += Jiuj / ni                            # q3,q4: meaned over all i≠j pairs
                acc[3] += Jiuj * Jjui / (ni * nj)
            cnt += 1; i_contrib = True
        if i_contrib:
            nis += 1
    dvp = cnt if cnt else 1                                     # # off-diagonal pairs (q3,q4)
    dvi = nis if nis else 1                                     # # samples i with a real overlap (q1,q2: mean_i Σ_j)
    out["xproj"] = {"min": [accmin[0] / dvi, accmin[1] / dvi, accmin[2] / dvp, accmin[3] / dvp],
                    "max": [accmax[0] / dvi, accmax[1] / dvi, accmax[2] / dvp, accmax[3] / dvp]}
    return out


SEC13_KS = (2, 5)   # §13 top⊕bottom ranks k₀ compared against the exact reference
SEC13_DOMS = ("ndist", "ijeq", "diag")   # i≠j≠k ; i=j≠k ; i=j=k  (the three (i,j,k) classes plotted)


def _sec13_stats(cur, prev, ref, lr):
    """§13 — approximations G1/G2/G3 of the exact reference J_iᵀQ_jJ_k·r_k, from per-sample Lanczos eigenpairs.
    For triple (i,j,k) and rank k₀ (D=2k₀ top⊕bottom dirs x,y,z of Q_i,Q_j,Q_k):
      G1 = Σ_{x,y,z} r_{k,t}σ_{j,y}·(1+r_{i,t-1}ησ_{i,x})cos_{(i,x)(j,y)}J'(x)_{i,t-1}·(1+r_{k,t-1}ησ_{k,z})cos_{(j,y)(k,z)}J'(z)_{k,t-1}
      G2 = Σ_{y,z}   r_{k,t}σ_{j,y}·(J_{i,t}·u_{j,y})·(1+r_{k,t-1}ησ_{k,z})cos_{(j,y)(k,z)}J'(z)_{k,t-1}      (i-side → J_{i,t}·u_{j,y})
      G3 = Σ_{x,y}   r_{k,t}σ_{j,y}·(1+r_{i,t-1}ησ_{i,x})cos_{(i,x)(j,y)}J'(x)_{i,t-1}·(J_{k,t}·u_{j,y})       (k-side → J_{k,t}·u_{j,y})
    Q_j (σ_{j,y}, u_{j,y}) is taken at step t; Q_i and Q_k (σ, u, J', r) at step t-1; r_{k,t} & σ_{j,y} at t. cos_{(a)(b)}
    are signed dot products of unit eigenvectors (cross-time for i↔j and j↔k). J'(x)_{i,t-1}=⟨J_{i,t-1},u_{i,x}^{t-1}⟩.
    cur/prev = {TV,TW,BV,BW,r,Jg}. ref=(N,N,N) exact J_iᵀQ_jJ_k·r_k at t. Returns mean/std over ALL (i,j,k) and over
    i≠j≠k for ref and each (G,k₀). MIRRORS eos_lab.linalg.sec13_stats / index.html sec13Stats."""
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


def _sec14_payload(prev, cur, lr, grid3dcap, rhist):
    """§14 — per-triplet (i,j,k) decomposition of Tr(ΔNTK)=Σ_{ℓ,p} a·b·c·d·e, with (gradient-gauged eigvecs
    u of the per-sample function Hessians Q): a=u_{i,ℓ}·u_{j,p}; b=σ_{i,ℓ}(J_i·u_{i,ℓ}); c=1+ησ_{j,p}r_j;
    d=u_{j,p}·J_k; e=r_k.
      TIME-INDEXING (t+1 = the CURRENT iterate): the eigenpairs u,σ and r_j (in c) and J_k (in d) are at the
      PREVIOUS eig-tick t; only J_i (in b) and r_k (in e) are at the current iterate t+1. So §14 needs BOTH ticks
      (prev,cur each = {TV,TW,BV,BW,r,Jg}) and SKIPS its first eig-tick (no prev), like §13. The full term T is
      gauge-invariant; eigvecs are gradient-gauged by the PREVIOUS gradient (⟨J^t_i,u⟩≥0) so the sub-products
      a, b·d, a·e are reproducible across backends.
    For rank-2y (y top⁺ ⊕ y bottom⁻ eigenpairs) the (2y)² terms are SORTED signed-descending per cell. Panels per
    y∈{1,2,5}: AGG=[max, min, Σtop-y, Σbot-y, Σall]; MAX/MIN=[a, b·d, a·b·d, a·b·d·e, a·e] at the argmax/argmin
    (ℓ,p). Panel 10 (eq-③) is UNCHANGED — it uses the CURRENT-tick eigenpairs (σ_{j,p} at t+1) and the residual
    HISTORY: ratio ∏_{z=1}^{S}(1+ησ_{j,p}r_{j,t-S+z}) (S=y), for {first,last} element × y. Each grid is packed
    (sparse + diagonal + μ/μ⁺/μ⁻). Gated N≤grid3dcap (N³ like §11). MIRRORS index.html sec14Payload /
    eos_lab.linalg.sec14_payload."""
    TVp, TWp, BVp, BWp, rp, Jgp = prev["TV"], prev["TW"], prev["BV"], prev["BW"], prev["r"], prev["Jg"]
    TVc, TWc, BVc, BWc, rc, Jgc = cur["TV"], cur["TW"], cur["BV"], cur["BW"], cur["r"], cur["Jg"]
    N, K, p = int(TVp.shape[0]), int(TVp.shape[1]), int(TVp.shape[2])
    out = {"M": N, "do3d": False}
    if N > grid3dcap:
        return out
    out["do3d"] = True
    dev = TVp.device
    dg = torch.arange(N, device=dev)

    def pack14(T):                                       # sparse subset + full diagonal + μ / μ⁺ / μ⁻
        v, idx = _g3d_pack(T.reshape(-1))
        f = T.reshape(-1); pos = f[f > 0]; neg = f[f < 0]
        return {"v": v, "idx": idx, "d": T[dg, dg, dg].detach().cpu().tolist(), "mn": float(f.mean()),
                "mp": float(pos.mean()) if pos.numel() else 0.0, "mng": float(neg.mean()) if neg.numel() else 0.0}

    def build(TVx, TWx, BVx, BWx, Jgg, Jgb, rcc, Jgd, ree, y):
        """rank-2y gauged factors with per-factor TIME sources: eigvecs/σ from (TVx,TWx,BVx,BWx); gauge gradient
        Jgg; b's J_i = Jgb; c's r_j = rcc; d's J_k = Jgd; e's r_k = ree. Returns a (N,N,D,D), b (N,D), d (N,D,N),
        e (N,), W (N,D σ), the full term T (N,N,N,D,D), D, kk."""
        kk = max(1, min(y, K)); D = 2 * kk
        E = torch.cat([TVx[:, :kk, :], BVx[:, :kk, :]], dim=1)        # (N,D,p) eigenvectors
        W = torch.cat([TWx[:, :kk], BWx[:, :kk]], dim=1)              # (N,D) signed eigenvalues σ
        gE = torch.einsum('ip,idp->id', Jgg, E)
        E = E * torch.where(gE >= 0, torch.ones_like(gE), -torch.ones_like(gE)).unsqueeze(2)   # GRADIENT GAUGE (prev grad)
        vE = E.norm(dim=2) > 0.5                                     # (N,D) REAL (unit) vs spurious (0) eigvec mask
        a = torch.einsum('ilp,jmp->ijlm', E, E)                      # (N,N,D,D)  u_{i,ℓ}·u_{j,p}
        b = W * torch.einsum('ip,idp->id', Jgb, E)                   # (N,D)      σ_{i,ℓ}(J_i·u_{i,ℓ}) ; J_i ← Jgb
        c = 1.0 + (lr / N) * W * rcc.view(N, 1)                      # (N,D)  1+(η/N)σ_{j,p}r_j ; r_j ← rcc (η/N = 1/N GD step)
        d = torch.einsum('jmp,kp->jmk', E, Jgd)                      # (N,D,N)    u_{j,p}·J_k ; J_k ← Jgd
        e = ree                                                      # (N,)       r_k ← ree
        T = (a.unsqueeze(2) * b.view(N, 1, 1, D, 1) * c.view(1, N, 1, 1, D)
             * d.permute(0, 2, 1).reshape(1, N, N, 1, D) * e.view(1, 1, N, 1, 1))   # (N,N,N,D,D)
        return a, b, d, e, W, T, D, kk, vE

    def argsel(Tf, D):                                   # FIRST (ℓ,p) on ties (flat ℓ·D+p order) → cross-backend parity
        ar = torch.arange(D * D, device=dev).view(1, 1, 1, D * D); big = torch.full((), D * D, device=dev, dtype=ar.dtype)
        imax = torch.where(Tf == Tf.amax(3, keepdim=True), ar, big).amin(3).clamp_(max=D * D - 1)   # clamp: NaN cell ⇒ no match ⇒ big
        imin = torch.where(Tf == Tf.amin(3, keepdim=True), ar, big).amin(3).clamp_(max=D * D - 1)
        return imax, imin

    inf = float("inf")
    def modemask(vE, D):                                 # (N,N,N,D²): mode (ℓ,p) REAL iff u_{i,ℓ} & u_{j,p} both real (∀k)
        return (vE.view(N, 1, D, 1) & vE.view(1, N, 1, D)).reshape(N, N, D * D).unsqueeze(2).expand(N, N, N, D * D)
    def finite(x):                                       # drop the ±inf left in all-spurious cells (no real mode) → 0
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x))

    p14 = {}; sig_sel = {"first": {}, "last": {}}
    for y in SEC14_YS:
        # ---- panels 1-9: u,σ,r_j(c),J_k(d) at t (prev); J_i(b)=cur, r_k(e)=cur; gauge = prev gradient ----
        a, b, d, e, W, T, D, kk, vE = build(TVp, TWp, BVp, BWp, Jgp, Jgc, rp, Jgp, rc, y)
        Tf = T.reshape(N, N, N, D * D)
        vm = modemask(vE, D)                             # SPURIOUS modes (zero-norm eigvec) excluded from sort/select
        Tmax = Tf.masked_fill(~vm, -inf); Tmin = Tf.masked_fill(~vm, inf)   # spurious → ∓inf so they're never max/min
        Tsx, _ = torch.sort(Tmax, dim=3, descending=True)   # real desc, spurious(-inf) last
        Tsn, _ = torch.sort(Tmin, dim=3, descending=False)  # real asc,  spurious(+inf) last
        agg = [finite(Tsx[..., 0]), finite(Tsn[..., 0]),    # max / min over REAL modes
               finite(Tsx[..., :kk]).sum(3), finite(Tsn[..., :kk]).sum(3), Tf.sum(3)]   # Σtop-y / Σbot-y (real); Σall (T=0 for spurious)
        imax = argsel(Tmax, D)[0]; imin = argsel(Tmin, D)[1]   # argmax/argmin restricted to REAL modes

        def subprods(idx):                               # 5 sub-products at the per-cell selected (ℓ*,p*)
            lst = idx // D; mst = idx % D
            a_sel = a.reshape(N, N, D * D).unsqueeze(2).expand(N, N, N, D * D).gather(3, idx.unsqueeze(3)).squeeze(3)
            b_sel = b.view(N, 1, 1, D).expand(N, N, N, D).gather(3, lst.unsqueeze(3)).squeeze(3)
            d_sel = d.permute(0, 2, 1).reshape(1, N, N, D).expand(N, N, N, D).gather(3, mst.unsqueeze(3)).squeeze(3)
            e_sel = e.view(1, 1, N).expand(N, N, N)
            return [a_sel, b_sel * d_sel, a_sel * b_sel * d_sel, a_sel * b_sel * d_sel * e_sel, a_sel * e_sel]
        sub_max = subprods(imax); sub_min = subprods(imin)
        p14[str(y)] = {"agg": [pack14(x) for x in agg], "max": [pack14(x) for x in sub_max],
                       "min": [pack14(x) for x in sub_min]}

        # ---- panel 10 (eq-③): all-CURRENT term → its argmax/argmin REAL mode + current σ_{j,p} ----
        _, _, _, _, Wc, Tc, Dc, _, vEc = build(TVc, TWc, BVc, BWc, Jgc, Jgc, rc, Jgc, rc, y)
        Tcf = Tc.reshape(N, N, N, Dc * Dc); vmc = modemask(vEc, Dc)
        imax_c = argsel(Tcf.masked_fill(~vmc, -inf), Dc)[0]; imin_c = argsel(Tcf.masked_fill(~vmc, inf), Dc)[1]
        Wjc = Wc.view(1, N, 1, Dc).expand(N, N, N, Dc)               # σ_{j,p} at t+1, selected by the current term
        sig_sel["first"][str(y)] = Wjc.gather(3, (imax_c % Dc).unsqueeze(3)).squeeze(3)
        sig_sel["last"][str(y)] = Wjc.gather(3, (imin_c % Dc).unsqueeze(3)).squeeze(3)
    out["p14"] = p14

    # panel 10 — multi-step ratio over the residual history; order: y∈{1,2,5} for FIRST element, then for LAST.
    ratios = []
    for sel in ("first", "last"):
        for y in SEC14_YS:
            S = y
            if len(rhist) < S:
                ratios.append(None); continue
            sig = sig_sel[sel][str(y)]
            prod = torch.ones(N, N, N, device=dev, dtype=DTYPE)
            for rt in rhist[-S:]:
                prod = prod * (1.0 + (lr / N) * sig * rt.view(1, N, 1))   # η/N (1/N-normalized GD step; matches §13)
            ratios.append(pack14(prod))
    out["ratio"] = ratios
    return out


_DEPS = 0.3     # default additive-ε strength (UI hyperparameter `divreg`). divergence = (D²−2Σ)/(D² + sign(D²)·ε·scale), scale =
def _divreg(num, den, scale, eps=_DEPS):   # MAGNITUDE of the 2nd-difference 2(Σ|term|). eps→0 ⇒ exactly the defined (D²−2Σ)/D² (spiky at D²→0);
    return num / (den + math.copysign(eps * scale, den)) * 100.0   # larger eps softens the 0/0 spike at ‖J‖²/σ₁ inflections (eps≳1 ⇒ ≈ residual/scale). User-tunable.


def _sec15_stats(hvp1, hvp2, Jt, Jtm1, Jtm2, rtm1, rtm2, lr, N, diveps=_DEPS):
    """§15 — 2nd-difference decomposition of ‖J‖²_F (panel 1) and σ₁ (panel 2). hvp1/hvp2 give {Q_{t-1,k}·v}_k /
    {Q_{t-2,k}·v}_k; Jt/Jtm1/Jtm2 are (N,p) per-sample GT-gradients; rtm1/rtm2 (N,). MIRRORS eos_lab.sec15_stats.
      I  = c²·g_{t−1}ᵀ S_{t−1} g_{t−1}, S=Σ_k Q²       II = c²·Σ_k ∇f_{t,k}ᵀ Q_{t−1,k} Q̃ g_{t−2}, Q̃=Σⱼ r_{t−1,j}Q_{t−2,j}
      III= c²·Σ_k ∇f_{t,k}ᵀ Q_{t−1,k}(J_{t−1}J_{t−2}ᵀJ_{t−2}r_{t−2})   A = c²·(J_{t−1}ᵀSJ_{t−1})(I−cJ_{t−2}ᵀJ_{t−2})
      IV/V/VI/B: σ₁ analogs — S→(Q̄ᵘ)², Q̄ᵘ=Σ_k u_{t,1,k}Q_{t−1,k}, trace-tie→u_{t,1} projection. c=η/N."""
    dt, dev = Jt.dtype, Jt.device
    c = lr / N
    c2 = c * c
    Ne = Jt.shape[0]                                          # effective samples = N·d_out (A,B are Ne×Ne); c uses N input samples

    def estats(M):                                            # real eigenvalues of N×N M (A,B are PSD·sym ⇒ real)
        e = torch.linalg.eigvals(M).real
        es = torch.sort(e, descending=True).values
        k = min(3, es.numel())
        top = es[:k].tolist()
        bot = torch.sort(e).values[:k].tolist()
        st = [float(e.min()), float(e.max()), float(e.mean()),
              float(torch.quantile(e, 0.25)), float(torch.quantile(e, 0.75))]
        return top, bot, st, es.tolist()                      # full spectrum (desc) for the SLQ density (plot 5)

    g1 = Jtm1.t() @ rtm1; g2 = Jtm2.t() @ rtm2
    Ktm2 = Jtm2 @ Jtm2.t()
    Imat = torch.eye(Ne, dtype=dt, device=dev)

    H1g1 = hvp1(g1)
    I = c2 * float((H1g1 * H1g1).sum())
    W = torch.stack([hvp1(Jtm1[i]) for i in range(Ne)])        # (N_i,N_k,p)
    JSJ = torch.einsum('ikp,lkp->il', W, W)
    # (*) term so I+II = r_{t−1}ᵀ A r_{t−2}: A_part2[j,i]=c²⟨ψ,Q_{t−2,j}∇f_{t−2,i}⟩, ψ=Σ_k Q_{t−1,k}∇f_{t,k}; M2[i,j]=Q_{t−2,j}∇f_{t−2,i}
    psi = torch.stack([hvp1(Jt[k])[k] for k in range(Ne)]).sum(0)
    M2 = torch.stack([hvp2(Jtm2[i]) for i in range(Ne)])
    A = c2 * (JSJ @ (Imat - c * Ktm2)) + c2 * torch.einsum('a,ija->ji', psi, M2)   # general (non-symmetric) ⇒ real-part eigenvalues
    Atop, Abot, Astat, Aeig = estats(A)

    a = rtm1 @ hvp2(g2)
    H1a = hvp1(a)
    II = c2 * float((Jt * H1a).sum())
    w = Ktm2 @ rtm2
    z = Jtm1.t() @ w
    H1z = hvp1(z)
    III = c2 * float((Jt * H1z).sum())

    # ── new Panels A: chained-contraction norms building II (A1 fwd, A2 bwd) and III (A3 fwd, A4 bwd) — MIRRORS eos_lab ──
    nrm = lambda v: float(v.norm())
    Qtp = rtm1 @ hvp2(psi)                                     # Q̃ ψ  (p,)   [1 extra HVP]
    M3a = Jtm2 @ Qtp; JtQt = torch.einsum('j,ijp->ip', rtm1, M2)
    p2A = Jtm1 @ psi; p3A = Jtm2.t() @ p2A; p4A = Jtm2 @ p3A
    npsi = nrm(psi); ng2 = nrm(g2)
    A1 = [npsi, nrm(Qtp), nrm(M3a), II]
    A2 = [ng2, nrm(a), II, abs(II), nrm(JtQt)]
    A3 = [npsi, nrm(p2A), nrm(p3A), nrm(p4A), III]
    A4 = [ng2, nrm(w), nrm(z), III, npsi]

    nJt = float((Jt * Jt).sum()); nJtm1 = float((Jtm1 * Jtm1).sum()); nJtm2 = float((Jtm2 * Jtm2).sum())
    D2 = nJt + nJtm2 - 2.0 * nJtm1
    divP = _divreg(-2.0 * (I + II - III) + D2, D2, 2.0 * (abs(I) + abs(II) + abs(III)), diveps)  # 2×: terms are ½∂²‖J‖²; predict D²≈2(I+II−III); →0 when theory holds. ε-scale = 2(|I|+|II|+|III|) (the D²-magnitude).

    Kt = Jt @ Jt.t()
    evt = _safe_eigh(Kt)
    sig_t = float(evt.eigenvalues[-1]); u1 = evt.eigenvectors[:, -1]
    sig_tm1 = float(_safe_eigvalsh(Jtm1 @ Jtm1.t())[-1])
    sig_tm2 = float(_safe_eigvalsh(Ktm2)[-1])

    Qu_g1 = u1 @ H1g1
    IV = c2 * float((Qu_g1 * Qu_g1).sum())
    b = u1 @ Jt
    V = c2 * float((b * (u1 @ H1a)).sum())
    VI = c2 * float((b * (u1 @ H1z)).sum())
    P = torch.einsum('k,ikp->ip', u1, W)
    Bm = c2 * ((P @ P.t()) @ (Imat - c * Ktm2))
    # (**) term so IV+V = r_{t−1}ᵀ B r_{t−2}: B_part2[j,i]=c²⟨Q̄ᵘb,Q_{t−2,j}∇f_{t−2,i}⟩, b=Σ_k u₁ₖ∇f_{t,k}
    Bm = Bm + c2 * torch.einsum('a,ija->ji', u1 @ hvp1(b), M2)
    Btop, Bbot, Bstat, Beig = estats(Bm)
    Dsig2 = sig_t + sig_tm2 - 2.0 * sig_tm1
    divS = _divreg(-2.0 * (IV + V - VI) + Dsig2, Dsig2, 2.0 * (abs(IV) + abs(V) + abs(VI)), diveps)  # 2×: terms are ½∂²σ₁; predict Dσ²≈2(IV+V−VI); →0 when theory holds. ε-scale = 2(|IV|+|V|+|VI|) (the Dσ²-magnitude).

    rec = {"I": I, "II": II, "III": III, "divP": divP, "Atop": Atop, "Abot": Abot, "Astat": Astat,
           "IV": IV, "V": V, "VI": VI, "divS": divS, "Btop": Btop, "Bbot": Bbot, "Bstat": Bstat,
           "D2": D2, "Dsig2": Dsig2, "Aeig": Aeig, "Beig": Beig,
           "A1": A1, "A2": A2, "A3": A3, "A4": A4,                       # new Panel A — II/III chained-contraction norms
           "Js": nJt + nJtm1 + nJtm2, "Ss": sig_t + sig_tm1 + sig_tm2}
    return rec, A, Bm, u1   # also expose A,B and the top NTK eigvec u₁ for panels 3/4 (Q̇≠0)


def _sec15_panelB(TV1, BV1, TW1, BW1, Jg):
    """§15 Panel B — per-sample ∇f_i projection ratio on the most-+ (u_i⁺) vs most-− (u_i⁻) eigvec of its own Q_i.
    ρ_i=|⟨∇f_i,u_i⁺⟩|/(|⟨∇f_i,u_i⁺⟩|+|⟨∇f_i,u_i⁻⟩|)·100; ρ'_i = same ×|eigenvalue|. Returns {mean,std} of each. MIRRORS eos_lab/browser."""
    pt = (Jg * TV1).sum(dim=1).abs(); pb = (Jg * BV1).sum(dim=1).abs()
    rho = 100.0 * pt / (pt + pb).clamp_min(1e-30)
    wt = pt * TW1.abs(); wb = pb * BW1.abs()
    rhow = 100.0 * wt / (wt + wb).clamp_min(1e-30)
    return {"mean": float(rho.mean()), "std": float(rho.std(unbiased=False)),
            "meanw": float(rhow.mean()), "stdw": float(rhow.std(unbiased=False))}


def _eig_stats15(M):   # real-part eigenvalue summary of a general N×N matrix → (top-3, bottom-3, [min,max,mean,q25,q75], full desc)
    e = torch.linalg.eigvals(M).real
    es = torch.sort(e, descending=True).values
    k = min(3, es.numel())
    return (es[:k].tolist(), torch.sort(e).values[:k].tolist(),
            [float(e.min()), float(e.max()), float(e.mean()), float(torch.quantile(e, 0.25)), float(torch.quantile(e, 0.75))],
            es.tolist())


def _sec15_panel34(hvp_at, th_tm2, Jt, Jtm1, Jtm2, rtm1, rtm2, u1, A, B, rec, lr, N, eps=1e-3, diveps=_DEPS):
    """Panels 3/4 (Q̇≠0): third-derivative terms VII/VIII + matrices A'=A+(VII), B'=B+(VIII). MIRRORS
    eos_lab.linalg.sec15_panel34 / index.html sec15Panel34. `hvp_at(theta,v)`={Q_k(theta)·v}_k at any theta;
    a directional FD of it gives T_{t-2,k}. Costs 2N²+2 HVP evals."""
    c = lr / N; c2 = c * c
    Ne = Jt.shape[0]                                          # effective samples = N·d_out (A',B' are Ne×Ne)
    g1 = Jtm1.t() @ rtm1; g2 = Jtm2.t() @ rtm2; b = u1 @ Jt
    dQg1 = (hvp_at(th_tm2 + eps * g2, g1) - hvp_at(th_tm2 - eps * g2, g1)) / (2.0 * eps)
    VII = c2 * float((Jt * dQg1).sum()); VIII = c2 * float((b * (u1 @ dQg1)).sum())
    dA = torch.zeros(Ne, Ne, dtype=Jt.dtype, device=Jt.device); dB = torch.zeros(Ne, Ne, dtype=Jt.dtype, device=Jt.device)
    for i in range(Ne):
        thp = th_tm2 + eps * Jtm2[i]; thm = th_tm2 - eps * Jtm2[i]
        for j in range(Ne):
            d = (hvp_at(thp, Jtm1[j]) - hvp_at(thm, Jtm1[j])) / (2.0 * eps)
            dA[j, i] = (Jt * d).sum(); dB[j, i] = (b * (u1 @ d)).sum()
    Aptop, Apbot, Apstat, Apeig = _eig_stats15(A + c2 * dA)
    Bptop, Bpbot, Bpstat, Bpeig = _eig_stats15(B + c2 * dB)
    D2 = rec["D2"]; Dsig2 = rec["Dsig2"]
    sJ = rec["I"] + rec["II"] - rec["III"] + VII; sS = rec["IV"] + rec["V"] - rec["VI"] + VIII
    return {"VII": VII, "VIII": VIII,
            "divP3": _divreg(-2.0 * sJ + D2, D2, 2.0 * (abs(rec["I"]) + abs(rec["II"]) + abs(rec["III"]) + abs(VII)), diveps),    # ε-scale = 2(|I|+|II|+|III|+|VII|)
            "divS4": _divreg(-2.0 * sS + Dsig2, Dsig2, 2.0 * (abs(rec["IV"]) + abs(rec["V"]) + abs(rec["VI"]) + abs(VIII)), diveps),  # ε-scale = 2(|IV|+|V|+|VI|+|VIII|)
            "Aptop": Aptop, "Apbot": Apbot, "Apstat": Apstat, "Apeig": Apeig,
            "Bptop": Bptop, "Bpbot": Bpbot, "Bpstat": Bpstat, "Bpeig": Bpeig}


# ===================== §16: curvature-aligned per-residual-sign optimizer (standalone) =====================
# MIRRORS eos_lab.sec16.sec16_run / index.html sec16Run. Runs in float64 (no catastrophic cancellation, but float64
# keeps the line-search/eig trajectory identical to the float64 eos_lab/browser backends). The server's shared Lanczos
# is DTYPE(float32)-typed, so §16 uses its own tiny float64 Lanczos below.
def _randvec16(p, seed):   # mulberry32+gauss start vector — IDENTICAL across server/eos_lab/browser (cross-backend §16 sync;
    rng = mulberry32(u32(int(seed)))                            # H̄'s clustered spectrum makes the eigenpair start-dependent at finite m)
    return torch.tensor([gauss(rng) for _ in range(p)], dtype=torch.float64, device=_dev())

def _shuffle16(N, seed):                                        # Fisher–Yates permutation of range(N) — IDENTICAL across backends
    rng = mulberry32(u32(int(seed))); idx = list(range(N))
    for i in range(N - 1, 0, -1):
        j = int(rng() * (i + 1)); idx[i], idx[j] = idx[j], idx[i]
    return idx

def _lanczos16_core(hvp, p, m, seed):
    q = _randvec16(p, seed); q = q / q.norm()
    Q = []; al = []; be = []
    qp = torch.zeros(p, dtype=torch.float64, device=_dev()); beta = torch.zeros((), dtype=torch.float64, device=_dev())
    for _ in range(min(p, m)):
        Q.append(q); w = hvp(q); a = (w @ q); al.append(float(a)); w = w - a * q - beta * qp
        Qm = torch.stack(Q)
        for _ in range(2):
            w = w - Qm.t() @ (Qm @ w)
        beta = w.norm(); be.append(float(beta))
        if float(beta) < 1e-10:
            break
        qp = q; q = w / beta
    k = len(Q); T = torch.zeros(k, k, dtype=torch.float64)
    for i in range(k):
        T[i, i] = al[i]
    for i in range(k - 1):
        T[i, i + 1] = be[i]; T[i + 1, i] = be[i]
    return torch.stack(Q), T, k

_Eigh = collections.namedtuple("_Eigh", ["eigenvalues", "eigenvectors"])   # mirrors torch.return_types.eigh (.eigenvalues/.eigenvectors + unpacks)

def _eig_ramp(T):                                    # add a DISTINCT per-index diagonal ramp to break the repeated-eigenvalue
    k = T.shape[-1]                                  # degeneracy that makes LAPACK/cuSOLVER syevd fail (code 22 CPU / 69 CUDA) on
    sc = max(float(T.abs().max()), 1e-12) * 1e-10    # flat-region tridiagonals. diag_embed broadcasts over any leading (batch) dims.
    return T + torch.diag_embed(torch.arange(k, dtype=T.dtype, device=T.device) * sc)

def _safe_eigvalsh(T):                               # robust symmetric-eigenvalue solve (cifar2/cifar10 §16/§17 hit near-zero tridiagonals)
    if not bool(torch.isfinite(T).all()):
        T = torch.nan_to_num(T)
    try:
        return torch.linalg.eigvalsh(T)
    except torch._C._LinAlgError:
        Tj = _eig_ramp(T)
        try:
            return torch.linalg.eigvalsh(Tj)
        except torch._C._LinAlgError:
            import numpy as np
            return torch.tensor(np.linalg.eigvalsh(Tj.detach().cpu().numpy()), dtype=T.dtype, device=T.device)

def _safe_eigh(T):                                   # returns a torch.linalg.eigh-like result (both unpacking AND .eigenvalues/.eigenvectors work)
    if not bool(torch.isfinite(T).all()):
        T = torch.nan_to_num(T)
    try:
        return torch.linalg.eigh(T)
    except torch._C._LinAlgError:
        Tj = _eig_ramp(T)
        try:
            return torch.linalg.eigh(Tj)
        except torch._C._LinAlgError:
            import numpy as np
            w, V = np.linalg.eigh(Tj.detach().cpu().numpy())
            return _Eigh(torch.tensor(w, dtype=T.dtype, device=T.device), torch.tensor(V, dtype=T.dtype, device=T.device))

def _lanc16_vals(hvp, p, n, m, seed):                # top-n & bottom-n eigenVALUES (float64)
    _, T, k = _lanczos16_core(hvp, p, m, seed); mu = torch.sort(_safe_eigvalsh(T), descending=True).values
    return [float(mu[min(i, k - 1)]) for i in range(n)], [float(mu[max(0, k - 1 - i)]) for i in range(n)]

def _lanc16_ext(hvp, p, n, m, seed):                 # top-n & bottom-n (eigVEC, eigVAL) pairs (float64)
    Qm, T, k = _lanczos16_core(hvp, p, m, seed); mu, Sv = _safe_eigh(T); idx = torch.argsort(mu, descending=True)
    ritz = lambda col: Sv[:, col].to(dtype=torch.float64, device=Qm.device) @ Qm   # Sv (CPU eigh) → Qm's device for the matmul
    tv = [ritz(int(idx[min(i, k - 1)])) for i in range(n)]; bv = [ritz(int(idx[max(0, k - 1 - i)])) for i in range(n)]
    tval = [float(mu[int(idx[min(i, k - 1)])]) for i in range(n)]; bval = [float(mu[int(idx[max(0, k - 1 - i)])]) for i in range(n)]
    return tv, bv, tval, bval

def _sec16_run(th0, X, Y, lr, warmup, iters, neig, mlan, seed, tol, bgrid=0.1, ares=0.01,
               Xtest=None, Ytest=None, mode="eig", kdir=8):
    # project g₊ onto the top-kdir eigvecs of H̄ and g₋ onto the bottom-kdir (coefficient ⟨g,u_i⟩, NO σ/η), then α₁/α₂/β/s.
    # every trajectory advances by its own u_sec16; modes differ ONLY in how it is built — 'eig' (main curvature),
    # 'random'/'shuffle'/'randfix'/'eigfix' (the curvature baselines), 'gd' (u_sec16 = the line-searched gradient step).
    p = th0.numel(); N = X.shape[0]; Yf = Y.reshape(-1)
    M = Yf.numel()                                                    # EFFECTIVE samples = N·d_out (the residual / Jacobian-row dimension)
    Ytf = Ytest.reshape(-1) if Ytest is not None else None
    k = max(1, int(neig)); mg = max(1, int(round(1.0 / max(1e-9, bgrid)))); grid = [i / mg for i in range(mg + 1)]
    fwd = lambda th: _TL.model.forward(th, X).reshape(-1)             # flat (M,) over all N·d_out outputs
    resid = lambda th: Yf - fwd(th)                                    # y − f(x)   (M,)
    def lossv(th, mask=None):
        out = fwd(th)
        if mask is None:
            return float(_TL.loss.value(out, Yf, N))                  # full loss normalised by N input samples (matches the main GD)
        nn = int(mask.sum()); return float(_TL.loss.value(out[mask], Yf[mask], max(1, nn))) if nn else 0.0
    def losstest(th):
        if Xtest is None:
            return None
        return float(_TL.loss.value(_TL.model.forward(th, Xtest).reshape(-1), Ytf, Xtest.shape[0]))
    Hbar = lambda th: (lambda v: jac_hvp(th, X, v).mean(0))            # (1/N)Σ ∇²f_k·v
    lossH = lambda th: (lambda v: (gradL(th + EPS * v, X, Y)[0] - gradL(th - EPS * v, X, Y)[0]) / (2 * EPS))
    def metrics(th, sd):
        lHt, _ = _lanc16_vals(lossH(th), p, k, mlan, sd)
        fHt, fHb = _lanc16_vals(Hbar(th), p, k, mlan, sd + 7)
        return {"L": lossv(th), "Lt": losstest(th), "res": resid(th).detach().cpu().tolist(), "lH": lHt, "fHt": fHt, "fHb": fHb}
    def pack(real, look=None):
        keys = ("pos", "neg", "mean", "beta"); d = look or {}
        rec = {"L": {kk: (d[kk]["L"] if kk in d else None) for kk in keys},
               "Lt": {kk: (d[kk]["Lt"] if kk in d else None) for kk in keys},
               "lH": {kk: (d[kk]["lH"] if kk in d else None) for kk in keys},
               "fH": {kk: ({"top": d[kk]["fHt"], "bot": d[kk]["fHb"]} if kk in d else None) for kk in keys},
               "res": {kk: (d[kk]["res"] if kk in d else None) for kk in keys}}
        rec["L"]["beta"] = real["L"]; rec["Lt"]["beta"] = real["Lt"]; rec["lH"]["beta"] = real["lH"]
        rec["fH"]["beta"] = {"top": real["fHt"], "bot": real["fHb"]}; rec["res"]["beta"] = real["res"]
        return rec
    def line_search(th, direction, mask, lo=0.0, hi=10.0, ls_tol=ares):
        gr = (math.sqrt(5) - 1) / 2; a, b = lo, hi
        c = b - gr * (b - a); dd = a + gr * (b - a)
        fc = lossv(th + c * direction, mask); fdd = lossv(th + dd * direction, mask)
        while (b - a) > ls_tol:
            if fc < fdd:
                b, dd, fdd = dd, c, fc; c = b - gr * (b - a); fc = lossv(th + c * direction, mask)
            else:
                a, c, fc = c, dd, fdd; dd = a + gr * (b - a); fdd = lossv(th + dd * direction, mask)
        al = math.floor((a + b) / 2.0 / ares + 0.5) * ares     # snap to the `ares` grid (half-up; identical in JS), α≥0
        return al if lossv(th + al * direction, mask) <= lossv(th, mask) else 0.0
    th = th0.detach().clone()
    froz = [None, None]                                                # frozen (u₁,u₋₁) for 'randfix'/'eigfix' (captured at warmup-end)
    for it in range(int(warmup) + int(iters)):
        sd = (int(seed) + 1000 * it) & 0x7FFFFFFF
        if it < warmup:                                                # standard GD warmup: θ ← θ + (lr/N)·Jᵀr (r=y−f; ADDED)
            real = metrics(th, sd)
            upd = (lr / N) * (jac_cols(th, X)[0].t() @ resid(th))      # the warmup GD step (Panel 6: ‖update‖₂)
            yield {"i": it, "phase": "gd", "apos": None, "aneg": None, "beta": None, "scale": None, "tgd": None,
                   "loss": real["L"], "unorm": {"pos": None, "neg": None, "mean": None, "beta": float(upd.norm())}, **pack(real)}
            th = th + upd; continue
        r = resid(th); J = jac_cols(th, X)[0]
        if mode == "gd":                                              # GD baseline: u_sec16 IS the line-searched gradient step
            d_gd = (J.t() @ r) / N
            t_gd = line_search(th, d_gd, None); u_sec16 = t_gd * d_gd
            u_pos = u_neg = u_mean = u_sec16; u_beta = u_sec16        # the gradient direction has no ± decomposition
            look = {"pos": metrics(th + u_pos, sd + 1), "neg": metrics(th + u_neg, sd + 2),
                    "mean": metrics(th + u_mean, sd + 3), "beta": metrics(th + u_beta, sd + 4)}
            yield {"i": it, "phase": "sec16", "apos": t_gd, "aneg": t_gd, "beta": t_gd, "scale": t_gd, "tgd": t_gd,
                   "loss": look["beta"]["L"],
                   "unorm": {"pos": float(u_pos.norm()), "neg": float(u_neg.norm()),
                             "mean": float(u_mean.norm()), "beta": float(u_beta.norm())}, **pack(look["beta"], look)}
            th = th + u_beta
            if lossv(th) < tol:
                break
            continue
        kd = max(1, min(int(kdir), p // 2))                           # eigenvectors per side (top-kd & bottom-kd)
        if mode in ("random", "randfix"):                             # kd random unit vecs per side instead of eigvecs
            if mode == "randfix" and froz[0] is not None:             # Baseline C: reuse the warmup-end frozen random dirs
                Ut, Ub = froz[0], froz[1]
            else:
                Ut = [_randvec16(p, (sd ^ 0x16A1 ^ (i * 0x101)) & 0x7FFFFFFF) for i in range(kd)]
                Ub = [_randvec16(p, (sd ^ 0x16B1 ^ (i * 0x101)) & 0x7FFFFFFF) for i in range(kd)]
                Ut = [w / w.norm() for w in Ut]; Ub = [w / w.norm() for w in Ub]
                if mode == "randfix":
                    froz[0], froz[1] = Ut, Ub                         # freeze at warmup-end (first §16 iter)
        else:                                                         # 'eig' / 'eigfix' / 'shuffle': top-kd & bottom-kd eigvecs of H̄
            if mode == "eigfix" and froz[0] is not None:              # Baseline D: reuse the warmup-end frozen eigvec subspace
                Ut, Ub = froz[0], froz[1]
            else:
                tv, bv, tval, bval = _lanc16_ext(Hbar(th), p, kd, mlan, sd)
                Ut, Ub = tv, bv
                if mode == "eigfix":
                    froz[0], froz[1] = Ut, Ub
        if mode == "shuffle":                                         # Baseline B: permute ± membership over the M effective samples (counts preserved)
            n_pos = int((r > 0).sum()); n_neg = int((r < 0).sum()); sidx = _shuffle16(M, sd ^ 0x16B0)
            pos = torch.zeros(M, dtype=torch.bool, device=th.device); neg = torch.zeros(M, dtype=torch.bool, device=th.device)
            if n_pos:
                pos[torch.tensor(sidx[:n_pos], dtype=torch.long, device=th.device)] = True
            if n_neg:
                neg[torch.tensor(sidx[n_pos:n_pos + n_neg], dtype=torch.long, device=th.device)] = True
        else:
            pos, neg = r > 0, r < 0
        g_pos = (J[pos].t() @ r[pos]) if bool(pos.any()) else torch.zeros(p, dtype=th.dtype, device=th.device)
        g_neg = (J[neg].t() @ r[neg]) if bool(neg.any()) else torch.zeros(p, dtype=th.dtype, device=th.device)
        proj_pos = torch.zeros(p, dtype=th.dtype, device=th.device)    # Σ_i ⟨g₊,u_i⟩ u_i (g₊ onto top-kd eigvecs; NO σ/η)
        proj_neg = torch.zeros(p, dtype=th.dtype, device=th.device)    # Σ_i ⟨g₋,u₋i⟩ u₋i (g₋ onto bottom-kd eigvecs; NO σ/η)
        for i in range(kd):
            proj_pos = proj_pos + float(g_pos @ Ut[i]) * Ut[i]
            proj_neg = proj_neg + float(g_neg @ Ub[i]) * Ub[i]
        a_pos = line_search(th, proj_pos, pos); a_neg = line_search(th, proj_neg, neg)
        u_pos, u_neg = a_pos * proj_pos, a_neg * proj_neg; u_mean = 0.5 * (u_pos + u_neg)
        beta, scale = min(((bb, ss) for bb in grid for ss in grid),
                          key=lambda bs: lossv(th + bs[1] * (bs[0] * u_pos + (1 - bs[0]) * u_neg)))
        u_sec16 = scale * (beta * u_pos + (1 - beta) * u_neg)
        u_beta = u_sec16; tgd = 0.0                                    # advance by u_sec16 — the main run is PURE curvature (no GD fallback)
        look = {"pos": metrics(th + u_pos, sd + 1), "neg": metrics(th + u_neg, sd + 2),
                "mean": metrics(th + u_mean, sd + 3), "beta": metrics(th + u_beta, sd + 4)}
        yield {"i": it, "phase": "sec16", "apos": a_pos, "aneg": a_neg, "beta": beta, "scale": scale, "tgd": tgd,
               "loss": look["beta"]["L"],                              # Panel 6: ‖u_•‖₂ of each look-ahead update
               "unorm": {"pos": float(u_pos.norm()), "neg": float(u_neg.norm()),
                         "mean": float(u_mean.norm()), "beta": float(u_beta.norm())}, **pack(look["beta"], look)}
        th = th + u_beta                                               # ADVANCE — loss goes DOWN every iteration
        if lossv(th) < tol:
            break


def _sec16_chebyshev_testset(N, degree, dt):
    """Held-out chebyshev §16 test set: the N-1 MIDPOINTS of the training grid linspace(-1,1,N), T_degree labels."""
    xt = torch.linspace(-1.0, 1.0, max(int(N), 1), dtype=dt, device=_dev())
    xm = (xt[:-1] + xt[1:]) / 2.0 if int(N) > 1 else xt.clone()
    if int(degree) <= 0:
        y = torch.ones_like(xm)
    else:
        tm2, tm1 = torch.ones_like(xm), xm.clone()
        for _ in range(2, int(degree) + 1):
            tm2, tm1 = tm1, 2 * xm * tm1 - tm2
        y = tm1
    return xm.unsqueeze(1), y.unsqueeze(1)


def _ksparse_testset(N, nbits, k, seed, dt):
    """Held-out k-sparse parity test set: the SAME fixed subset S as training, fresh ±1 bits from a
    DISJOINT mulberry32 stream (seed*7919+99991). MIRRORS eos_lab.make_test_set / index.html sec16Holdout."""
    nb = max(1, int(nbits)); kk = max(1, min(int(k), nb))
    perm = _shuffle16(nb, (int(seed) ^ 0x4B5A11) & 0x7FFFFFFF); S = set(perm[:kk])
    rng = mulberry32(u32(int(seed) * 7919 + 99991))
    X = torch.empty(int(N), nb, dtype=dt, device=_dev()); Y = torch.empty(int(N), 1, dtype=dt, device=_dev())
    for i in range(int(N)):
        prod = 1.0
        for j in range(nb):
            b = 1.0 if rng() < 0.5 else -1.0
            X[i, j] = b
            if j in S:
                prod *= b
        Y[i, 0] = prod
    return X, Y


def _two_class_test_raw(raw, N, P):
    """Held-out 2-class SCALAR test set (classes {c2a,c2b} → ±1) from a (X_images, int_labels) split. float64."""
    import numpy as np
    Xa, Ya = raw; dev = _dev()
    ca, cb = int(P.get("c2a", 0)), int(P.get("c2b", 1))
    rng = np.random.RandomState((int(P["seed"]) & 0x7FFFFFFF) ^ 0x5151)
    ia = np.where(Ya == ca)[0]; ib = np.where(Ya == cb)[0]; rng.shuffle(ia); rng.shuffle(ib)
    na = int(N) // 2; nb = int(N) - na
    idx = np.concatenate([ia[:na], ib[:nb]])
    lab = np.concatenate([np.ones(min(na, len(ia)), np.float32), -np.ones(min(nb, len(ib)), np.float32)])
    perm = rng.permutation(len(idx)); idx = idx[perm]; lab = lab[perm]
    return (torch.tensor(Xa[idx], dtype=torch.float64, device=dev),
            torch.tensor(lab, dtype=torch.float64, device=dev).reshape(-1, 1))


def _sec16_holdout(dataset, N, P, inD, outD):
    """Held-out §16/§17 test set. Real held-out images for cifar/mnist (incl. the 2-class scalar variants), fresh
    iid Gaussian for synthetic/const. MIRRORS eos_lab.make_test_set / index.html sec16Holdout. float64. Returns
    (None, None) when ill-defined (fixed-input / sign-forced synthetic-const → no clean held-out split)."""
    import numpy as np
    dev = _dev()
    if dataset in ("cifar10", "cifar2"):
        raw = _load_cifar_test_raw()
        if raw is None:
            return None, None
        if dataset == "cifar2":
            return _two_class_test_raw(raw, N, P)
        Xa, Ya = raw; rng = np.random.RandomState((int(P["seed"]) & 0x7FFFFFFF) ^ 0x5151)
        idx = rng.permutation(Xa.shape[0])[:int(N)]
        Yt = torch.zeros(len(idx), 10, dtype=torch.float64, device=dev)
        Yt[torch.arange(len(idx)), torch.tensor(Ya[idx], dtype=torch.long, device=dev)] = 1.0
        return torch.tensor(Xa[idx], dtype=torch.float64, device=dev), Yt
    if dataset in ("mnist", "mnist2"):
        try:
            raw = _load_mnist_raw("test")
        except FileNotFoundError:
            return None, None
        if dataset == "mnist2":
            return _two_class_test_raw(raw, N, P)
        Xa, Ya = raw; rng = np.random.RandomState((int(P["seed"]) & 0x7FFFFFFF) ^ 0x5151)
        idx = rng.permutation(Xa.shape[0])[:int(N)]
        Yt = torch.zeros(len(idx), 10, dtype=torch.float64, device=dev)
        Yt[torch.arange(len(idx)), torch.tensor(Ya[idx], dtype=torch.long, device=dev)] = 1.0
        return torch.tensor(Xa[idx], dtype=torch.float64, device=dev), Yt
    if dataset == "ksparse":
        return _ksparse_testset(int(N), int(inD), int(P.get("ksparse", 3)), int(P["seed"]), torch.float64)
    if dataset not in ("synthetic", "const") or P.get("ssign", "off") != "off":
        return None, None
    if dataset == "synthetic" and P.get("fixedx", "0") == "1":
        return None, None
    trng = mulberry32(u32(int(P["seed"]) * 7919 + 99991))      # separate stream ⇒ disjoint from training (seed*7919+1)
    tgt = float(P["tgt"]); inStd = float(P["inputstd"]); dev = _dev()
    Xl = [[inStd * gauss(trng) for _ in range(int(inD))] for _ in range(int(N))]      # X drawn first, then Y (order matches make_test_set)
    if dataset == "const":
        cstd = max(0.0, float(P.get("cvar", 0.0))) ** 0.5
        Yl = [[abs(tgt) + cstd * gauss(trng) for _ in range(int(outD))] for _ in range(int(N))]
    else:
        Yl = [[tgt * gauss(trng) for _ in range(int(outD))] for _ in range(int(N))]
    return (torch.tensor(Xl, dtype=torch.float64, device=dev),
            torch.tensor(Yl, dtype=torch.float64, device=dev))


def _sec16_driver(th0, X, Y, lr, warmup, iters, neig, mlan, seed, tol, bgrid, ares, Xtest, Ytest, baselines, kdir=8):
    """§16 main ('eig', PURE curvature) and, if `baselines`, FIVE trajectories — 'random' (A), 'shuffle' (B),
    'randfix' (C: frozen random dirs), 'eigfix' (D: frozen H̄ eigvec dirs), 'gd' (E: the plain GD step) — in
    lockstep from the SAME θ₀. All six advance by their own u_sec16. rec['bA']/['bB']/['bC']/['bD']/['bE']."""
    gmain = _sec16_run(th0, X, Y, lr, warmup, iters, neig, mlan, seed, tol, bgrid, ares, Xtest, Ytest, "eig", kdir)
    if not baselines:
        yield from gmain
        return
    mk = lambda md: _sec16_run(th0, X, Y, lr, warmup, iters, neig, mlan, seed, -1.0, bgrid, ares, Xtest, Ytest, md, kdir)
    gA, gB, gC, gD, gE = mk("random"), mk("shuffle"), mk("randfix"), mk("eigfix"), mk("gd")
    sub = lambda r: {kk: r[kk] for kk in ("L", "Lt", "lH", "fH", "res", "unorm")}
    for rec in gmain:
        rec = dict(rec)
        rec["bA"] = sub(next(gA)); rec["bB"] = sub(next(gB))
        rec["bC"] = sub(next(gC)); rec["bD"] = sub(next(gD)); rec["bE"] = sub(next(gE))
        yield rec


def _sec17_run(th0, X, Y, lr, warmup, iters, neig, mlan, seed, tol, bgrid=0.1, ares=0.01,
               Xtest=None, Ytest=None, mode="eig", kdir=8):
    # §17 = §16 but PER-SAMPLE function Hessians Q_k=∇²f_k: r_k>0 → project r_k∇f_k onto the TOP-kdir eigvecs of Q_k,
    # r_k<0 → onto the BOTTOM-kdir (coefficient ⟨r_k∇f_k,u⟩, NO σ/η), then α₁/α₂/β/s. Main is PURE curvature; 'gd' baseline only.
    # MIRRORS eos_lab.sec17_run / index.html sec17Run. Q_k·v = jac_hvp(...)[k] (row k of the FD Jacobian-of-grad).
    p = th0.numel(); N = X.shape[0]; Yf = Y.reshape(-1)
    M = Yf.numel()                                                    # EFFECTIVE samples = N·d_out (per-sample-Q dimension)
    Ytf = Ytest.reshape(-1) if Ytest is not None else None
    dev = th0.device
    k = max(1, int(neig)); mg = max(1, int(round(1.0 / max(1e-9, bgrid)))); grid = [i / mg for i in range(mg + 1)]
    fwd = lambda th: _TL.model.forward(th, X).reshape(-1)
    resid = lambda th: Yf - fwd(th)
    def lossv(th, mask=None):
        out = fwd(th)
        if mask is None:
            return float(_TL.loss.value(out, Yf, N))
        nn = int(mask.sum()); return float(_TL.loss.value(out[mask], Yf[mask], max(1, nn))) if nn else 0.0
    def losstest(th):
        if Xtest is None:
            return None
        return float(_TL.loss.value(_TL.model.forward(th, Xtest).reshape(-1), Ytf, Xtest.shape[0]))
    Hbar = lambda th: (lambda v: jac_hvp(th, X, v).mean(0))            # Panel 3 diagnostic (same as §16)
    Qk = lambda th, kk: (lambda v: jac_hvp(th, X, v)[kk])              # per-sample Q_k·v
    lossH = lambda th: (lambda v: (gradL(th + EPS * v, X, Y)[0] - gradL(th - EPS * v, X, Y)[0]) / (2 * EPS))
    def metrics(th, sd):
        lHt, _ = _lanc16_vals(lossH(th), p, k, mlan, sd)
        fHt, fHb = _lanc16_vals(Hbar(th), p, k, mlan, sd + 7)
        return {"L": lossv(th), "Lt": losstest(th), "res": resid(th).detach().cpu().tolist(), "lH": lHt, "fHt": fHt, "fHb": fHb}
    def pack(real, look=None):
        keys = ("pos", "neg", "mean", "beta"); d = look or {}
        rec = {"L": {kk: (d[kk]["L"] if kk in d else None) for kk in keys},
               "Lt": {kk: (d[kk]["Lt"] if kk in d else None) for kk in keys},
               "lH": {kk: (d[kk]["lH"] if kk in d else None) for kk in keys},
               "fH": {kk: ({"top": d[kk]["fHt"], "bot": d[kk]["fHb"]} if kk in d else None) for kk in keys},
               "res": {kk: (d[kk]["res"] if kk in d else None) for kk in keys}}
        rec["L"]["beta"] = real["L"]; rec["Lt"]["beta"] = real["Lt"]; rec["lH"]["beta"] = real["lH"]
        rec["fH"]["beta"] = {"top": real["fHt"], "bot": real["fHb"]}; rec["res"]["beta"] = real["res"]
        return rec
    def line_search(th, direction, mask, lo=0.0, hi=10.0, ls_tol=ares):
        gr = (math.sqrt(5) - 1) / 2; a, b = lo, hi
        c = b - gr * (b - a); dd = a + gr * (b - a)
        fc = lossv(th + c * direction, mask); fdd = lossv(th + dd * direction, mask)
        while (b - a) > ls_tol:
            if fc < fdd:
                b, dd, fdd = dd, c, fc; c = b - gr * (b - a); fc = lossv(th + c * direction, mask)
            else:
                a, c, fc = c, dd, fdd; dd = a + gr * (b - a); fdd = lossv(th + dd * direction, mask)
        al = math.floor((a + b) / 2.0 / ares + 0.5) * ares
        return al if lossv(th + al * direction, mask) <= lossv(th, mask) else 0.0
    th = th0.detach().clone()
    froz = [None, None]                                               # frozen per-sample [Up, Um] for 'randfix'/'eigfix'
    for it in range(int(warmup) + int(iters)):
        sd = (int(seed) + 1000 * it) & 0x7FFFFFFF
        if it < warmup:                                               # standard GD warmup (identical to §16)
            real = metrics(th, sd)
            upd = (lr / N) * (jac_cols(th, X)[0].t() @ resid(th))
            yield {"i": it, "phase": "gd", "apos": None, "aneg": None, "beta": None, "scale": None, "tgd": None,
                   "loss": real["L"], "unorm": {"pos": None, "neg": None, "mean": None, "beta": float(upd.norm())}, **pack(real)}
            th = th + upd; continue
        r = resid(th); J = jac_cols(th, X)[0]                        # rows ∇f_k
        if mode == "gd":                                             # GD baseline: u_sec IS the line-searched gradient step (no per-sample work)
            d_gd = (J.t() @ r) / N; t_gd = line_search(th, d_gd, None); u_sec = t_gd * d_gd
            u_pos = u_neg = u_mean = u_beta = u_sec                  # the gradient direction has no ± decomposition
            look = {"pos": metrics(th + u_pos, sd + 1), "neg": metrics(th + u_neg, sd + 2),
                    "mean": metrics(th + u_mean, sd + 3), "beta": metrics(th + u_beta, sd + 4)}
            yield {"i": it, "phase": "sec17", "apos": t_gd, "aneg": t_gd, "beta": t_gd, "scale": t_gd, "tgd": t_gd,
                   "loss": look["beta"]["L"],
                   "unorm": {"pos": float(u_pos.norm()), "neg": float(u_neg.norm()),
                             "mean": float(u_mean.norm()), "beta": float(u_beta.norm())}, **pack(look["beta"], look)}
            th = th + u_beta
            if lossv(th) < tol:
                break
            continue
        psd = (sd ^ 0x17C0FFEE) & 0x7FFFFFFF                          # per-sample Lanczos/random seed base
        kd = max(1, min(int(kdir), p // 2))                          # eigvecs per side, per sample
        Up = []; Um = []                                             # per-sample lists of kd top / kd bottom eigvecs of Q_k
        for kk in range(M):
            skk = (psd + kk * 1009) & 0x7FFFFFFF
            hk = Qk(th, kk)
            if mode in ("random", "randfix"):                        # per-sample kd random dirs per side
                ut = [_randvec16(p, (skk ^ 0x17A1 ^ (i * 0x101)) & 0x7FFFFFFF) for i in range(kd)]
                um = [_randvec16(p, (skk ^ 0x17A2 ^ (i * 0x101)) & 0x7FFFFFFF) for i in range(kd)]
                Up.append([w / w.norm() for w in ut]); Um.append([w / w.norm() for w in um])
            else:                                                    # 'eig'/'eigfix'/'shuffle': per-sample top-kd/bottom-kd eigvecs of Q_k
                tv, bv, tval, bval = _lanc16_ext(hk, p, kd, mlan, skk); Up.append(list(tv)); Um.append(list(bv))
        if mode in ("randfix", "eigfix"):                            # Baselines C/D: freeze per-sample direction set at warmup-end
            if froz[0] is None:
                froz[0] = [[u.clone() for u in lst] for lst in Up]; froz[1] = [[u.clone() for u in lst] for lst in Um]
            Up, Um = froz[0], froz[1]
        if mode == "shuffle":                                        # Baseline B: permute ± membership over the M effective samples
            n_pos = int((r > 0).sum()); n_neg = int((r < 0).sum()); sidx = _shuffle16(M, sd ^ 0x16B0)
            pos = torch.zeros(M, dtype=torch.bool, device=dev); neg = torch.zeros(M, dtype=torch.bool, device=dev)
            if n_pos:
                pos[torch.tensor(sidx[:n_pos], dtype=torch.long, device=dev)] = True
            if n_neg:
                neg[torch.tensor(sidx[n_pos:n_pos + n_neg], dtype=torch.long, device=dev)] = True
        else:
            pos, neg = r > 0, r < 0
        proj_pos = torch.zeros(p, dtype=th.dtype, device=dev)        # Σ_{r_k>0} proj of r_k∇f_k onto Q_k's top-kd eigvecs (NO σ)
        proj_neg = torch.zeros(p, dtype=th.dtype, device=dev)        # Σ_{r_k<0} proj of r_k∇f_k onto Q_k's bottom-kd eigvecs (NO σ)
        for kk in range(M):
            if bool(pos[kk]):
                for i in range(kd):
                    proj_pos = proj_pos + (float(r[kk]) * float(J[kk] @ Up[kk][i])) * Up[kk][i]
            elif bool(neg[kk]):
                for i in range(kd):
                    proj_neg = proj_neg + (float(r[kk]) * float(J[kk] @ Um[kk][i])) * Um[kk][i]
        a_pos = line_search(th, proj_pos, pos); a_neg = line_search(th, proj_neg, neg)
        u_pos, u_neg = a_pos * proj_pos, a_neg * proj_neg; u_mean = 0.5 * (u_pos + u_neg)
        beta, scale = min(((bb, ss) for bb in grid for ss in grid),
                          key=lambda bs: lossv(th + bs[1] * (bs[0] * u_pos + (1 - bs[0]) * u_neg)))
        u_sec = scale * (beta * u_pos + (1 - beta) * u_neg)
        u_beta = u_sec; tgd = 0.0                                    # advance by u_sec — the main run is PURE curvature (no GD fallback)
        look = {"pos": metrics(th + u_pos, sd + 1), "neg": metrics(th + u_neg, sd + 2),
                "mean": metrics(th + u_mean, sd + 3), "beta": metrics(th + u_beta, sd + 4)}
        yield {"i": it, "phase": "sec17", "apos": a_pos, "aneg": a_neg, "beta": beta, "scale": scale, "tgd": tgd,
               "loss": look["beta"]["L"],
               "unorm": {"pos": float(u_pos.norm()), "neg": float(u_neg.norm()),
                         "mean": float(u_mean.norm()), "beta": float(u_beta.norm())}, **pack(look["beta"], look)}
        th = th + u_beta
        if lossv(th) < tol:
            break


def _sec17_driver(th0, X, Y, lr, warmup, iters, neig, mlan, seed, tol, bgrid, ares, Xtest, Ytest, baselines, kdir=8):
    """§17 main ('eig', per-sample Q_k, PURE curvature) and, if `baselines`, FIVE trajectories — 'random' (A),
    'shuffle' (B), 'randfix' (C), 'eigfix' (D), 'gd' (E: the plain GD step) — in lockstep from the SAME θ₀,
    attaching rec['bA']/['bB']/['bC']/['bD']/['bE']. All six advance by their own u_sec16. MIRRORS _sec16_driver."""
    gmain = _sec17_run(th0, X, Y, lr, warmup, iters, neig, mlan, seed, tol, bgrid, ares, Xtest, Ytest, "eig", kdir)
    if not baselines:
        yield from gmain
        return
    mk = lambda md: _sec17_run(th0, X, Y, lr, warmup, iters, neig, mlan, seed, -1.0, bgrid, ares, Xtest, Ytest, md, kdir)
    gA, gB, gC, gD, gE = mk("random"), mk("shuffle"), mk("randfix"), mk("eigfix"), mk("gd")
    sub = lambda r: {kk: r[kk] for kk in ("L", "Lt", "lH", "fH", "res", "unorm")}
    for rec in gmain:
        rec = dict(rec)
        rec["bA"] = sub(next(gA)); rec["bB"] = sub(next(gB))
        rec["bC"] = sub(next(gC)); rec["bD"] = sub(next(gD)); rec["bE"] = sub(next(gE))
        yield rec


def _opt_dir(model, g, opt):
    """Update DIRECTION from the raw gradient g (NO momentum). The actual θ step is θ ← θ − lr·dir.
      gd       → g                       (plain full-batch gradient descent)
      sign     → sign(g)                 (signed GD, elementwise)
      spectral → Muon-style: each weight MATRIX W gets svd(∇W)=UΣVᵀ → step UVᵀ (steepest descent under the
                 spectral norm); biases / non-matrix params fall back to sign(g).
    Only the actual trajectory changes — the §9/§10 theory recomputes its own GD Δθ and is untouched."""
    if opt == "sign":
        return torch.sign(g)
    if opt == "spectral":
        d = torch.sign(g)                       # default for biases / non-matrix params
        spec = getattr(model, "spec", None)
        if spec is not None:                    # MLP: layer ℓ weight is a din×dout contiguous block of θ
            for layer in spec:
                din, dout, wOff = layer[0], layer[1], layer[3]
                G = g[wOff:wOff + din * dout].view(din, dout)
                U, _, Vh = torch.linalg.svd(G, full_matrices=False)
                d[wOff:wOff + din * dout] = (U @ Vh).reshape(-1)
        return d
    return g                                    # gd (default)
# Each /run is assigned a device (auto least-busy across DEVICE_POOL). RUN_TOKEN is PER-DEVICE, so runs on
# different GPUs coexist; a newer run on the SAME device bumps that device's token and the older one stops.
# _DEV_LOAD counts active runs per device for the least-busy pick. All three are guarded by _DEV_LOCK.
_DEV_LOCK = threading.Lock()
_DEV_LOAD = {}                 # device-string -> number of active runs
RUN_TOKEN = {}                 # device-string -> monotonic token
_TL = threading.local()       # per-request model/loss + assigned device (ThreadingHTTPServer: one thread per /run)
CIFAR_DIR = None               # CLI override for the CIFAR-10 raw-batches directory (auto-detected if None)

def _dev():
    """Device assigned to the current /run thread; falls back to the default _dev() off-thread."""
    return getattr(_TL, "device", DEVICE)

def acquire_device():
    """Pick the least-busy device in the pool for a new run and claim a fresh per-device token."""
    with _DEV_LOCK:
        pool = DEVICE_POOL or [DEVICE]
        dev = min(pool, key=lambda d: (_DEV_LOAD.get(str(d), 0), pool.index(d)))
        k = str(dev)
        _DEV_LOAD[k] = _DEV_LOAD.get(k, 0) + 1
        RUN_TOKEN[k] = RUN_TOKEN.get(k, 0) + 1
        return dev, RUN_TOKEN[k]

def release_device(dev):
    with _DEV_LOCK:
        k = str(dev)
        _DEV_LOAD[k] = max(0, _DEV_LOAD.get(k, 0) - 1)


# ===================== mulberry32 (exact JS port; data + init) =====================
def u32(x):
    return x & 0xFFFFFFFF


def i32(x):
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x >= 0x80000000 else x


def imul(a, b):
    return i32((u32(a) * u32(b)) & 0xFFFFFFFF)


def xor32(a, b):
    return i32(u32(a) ^ u32(b))


def or32(a, b):
    return u32(a) | u32(b)


def mulberry32(seed):
    st = {"a": i32(seed)}

    def rnd():
        a = i32(st["a"] + 0x6D2B79F5)
        st["a"] = a
        t = imul(xor32(a, u32(a) >> 15), or32(1, a))
        t = xor32(i32(t + imul(xor32(t, u32(t) >> 7), or32(61, t))), t)
        return u32(xor32(t, u32(t) >> 14)) / 4294967296.0

    return rnd


def gauss(rng):
    u = 0.0
    v = 0.0
    while u == 0:
        u = rng()
    while v == 0:
        v = rng()
    return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)


# ===================== model (torch, multi-output, no autograd) =====================
def build_spec(inDim, width, depth, act, useBias, outDim):
    spec = []
    off = 0
    din = inDim
    for _ in range(depth):
        dout = width
        wOff = off
        off += din * dout
        bOff = off if useBias else -1
        if useBias:
            off += dout
        spec.append((din, dout, act, wOff, bOff))
        din = dout
    wOff = off
    off += din * outDim
    bOff = off if useBias else -1
    if useBias:
        off += outDim
    spec.append((din, outDim, "linear", wOff, bOff))
    return spec, off


def actf(name, z):
    if name == "tanh":
        return torch.tanh(z)
    if name == "elu":
        return torch.where(z > 0, z, torch.expm1(z))
    if name == "quadratic":
        return z * z
    return z


def actd(name, z):
    if name == "tanh":
        t = torch.tanh(z)
        return 1 - t * t
    if name == "elu":
        return torch.where(z > 0, torch.ones_like(z), torch.exp(z))
    if name == "quadratic":
        return 2 * z
    return torch.ones_like(z)


def fwd(th, spec, X):
    a = X
    caches = []
    for (din, dout, act, wOff, bOff) in spec:
        W = th[wOff:wOff + din * dout].view(din, dout)
        z = a @ W
        if bOff >= 0:
            z = z + th[bOff:bOff + dout]
        caches.append((a, z, act, din, dout, wOff, bOff))
        a = actf(act, z)
    return a, caches


def bwd(th, spec, caches, dO):
    g = torch.zeros(th.shape[0], dtype=th.dtype, device=th.device)
    dA = dO
    for (a, z, act, din, dout, wOff, bOff) in reversed(caches):
        dZ = dA * actd(act, z)
        g[wOff:wOff + din * dout] = (a.t() @ dZ).reshape(-1)
        if bOff >= 0:
            g[bOff:bOff + dout] = dZ.sum(0)
        W = th[wOff:wOff + din * dout].view(din, dout)
        dA = dZ @ W.t()
    return g


def bwd_batch(th, spec, caches, dO):
    """Backward for a BATCH of B seed vectors at once. dO: (B, N, outD) → G: (B, p).
    Same math as bwd but vectorized over the batch (and over samples), so the whole
    Jacobian is one set of matmuls instead of B Python-level backward passes."""
    B = dO.shape[0]
    g = torch.zeros(B, th.shape[0], dtype=th.dtype, device=th.device)
    dA = dO                                                  # (B, N, dout)
    for (a, z, act, din, dout, wOff, bOff) in reversed(caches):
        dZ = dA * actd(act, z)                               # (B,N,dout) * (N,dout) broadcast
        g[:, wOff:wOff + din * dout] = torch.einsum('ni,bno->bio', a, dZ).reshape(B, -1)
        if bOff >= 0:
            g[:, bOff:bOff + dout] = dZ.sum(1)               # (B, dout)
        W = th[wOff:wOff + din * dout].view(din, dout)
        dA = dZ @ W.t()                                      # (B,N,dout) @ (dout,din) -> (B,N,din)
    return g


def gradF(th, X):
    """∇_θ Σf — gradient of the summed output. Returns (g, out)."""
    return _TL.model.vjp(th, X, lambda out: torch.ones_like(out))


def gradL(th, X, Y):
    """∇_θ L — loss gradient (loss-aware cotangent). Returns (g, out)."""
    N = X.shape[0]
    return _TL.model.vjp(th, X, lambda out: _TL.loss.resid_cotangent(out, Y, N))


def gradW(th, X, c):
    """∇_θ Σ(c⊙f) — backward with an arbitrary output cotangent c (N,oc). Returns g."""
    return _TL.model.vjp(th, X, c)[0]


def init_kind_scale(scheme, init, fan_in, fan_out, is_readout):
    """Weight-draw spec for `scheme`: ('normal', std) or ('uniform', half-range). MIRRORS
    eos_lab.models.init_kind_scale / index.html initKindScale. The named theoretical schemes
    (mup/xavier_*) use their CANONICAL variance and IGNORE `init`; only default (scale=init/√fan_in)
    and custom (std=init) use the `init_scale` knob."""
    fi = max(int(fan_in), 1)
    fo = max(int(fan_out), 1)
    if scheme == "xavier_normal":
        return ("normal", math.sqrt(2.0 / (fi + fo)))                # canonical Glorot; init_scale ignored
    if scheme == "xavier_uniform":
        return ("uniform", math.sqrt(6.0 / (fi + fo)))               # canonical Glorot uniform; init_scale ignored
    if scheme == "mup":
        return ("normal", 1.0 / fi if is_readout else 1.0 / math.sqrt(fi))   # canonical muP; init_scale ignored
    if scheme == "custom":
        return ("normal", float(init))
    return ("normal", init / math.sqrt(fi))      # "default" — scale/√fan_in (init_scale = the scale)


def init_theta(spec, p, initScale, seed, scheme="default"):
    rng = mulberry32(u32(seed))
    th = [0.0] * p
    nl = len(spec)
    for li, (din, dout, act, wOff, bOff) in enumerate(spec):
        kind, sc = init_kind_scale(scheme, initScale, din, dout, li == nl - 1)
        for i in range(din * dout):
            th[wOff + i] = (gauss(rng) * sc) if kind == "normal" else ((rng() * 2.0 - 1.0) * sc)
    return torch.tensor(th, dtype=DTYPE, device=_dev())


# ===================== model / loss abstraction =====================
# Every diagnostic is re-expressed in terms of MODEL.forward(th,X)->out (N,oc) and
# MODEL.vjp(th,X,cot)->(g,out) [g = ∇_θ Σ(cot⊙out)], so the whole matrix-free suite is
# architecture-agnostic. `cot` is a tensor (N,oc) OR a callable out->(N,oc). The MLP keeps its
# exact hand-written fwd/bwd (browser-identical numerics); new architectures (GPU-server-only)
# implement forward functionally from flat-th slices and use autograd (exact double-backward HVPs).
# INVARIANT: forward returns (N,oc) and vjp consumes cot as (N,oc), row-major sample-major, for
# every architecture — matching out.reshape(-1), i*oc+c, eye(M).reshape(M,N,oc) used downstream.
class Model:
    exact_hvp = False
    in_shape = None
    oc = 1
    p = 0

    def forward(self, th, X):
        raise NotImplementedError

    def vjp(self, th, X, cot):
        raise NotImplementedError

    def jac_cols(self, th, X):
        """Per-output Jacobian J (M,p) (row a = ∇f_a) + flat out (M,). Default = batched vjp loop."""
        out = self.forward(th, X)
        N, oc = out.shape
        M = N * oc
        rows = []
        for a in range(M):
            dO = torch.zeros(N, oc, dtype=th.dtype, device=th.device)
            dO.reshape(-1)[a] = 1.0
            rows.append(self.vjp(th, X, dO)[0])
        return torch.stack(rows), out.reshape(-1)

    def init_theta(self, seed, initScale):
        raise NotImplementedError


class MlpModel(Model):
    """Dense MLP — delegates to the hand-written fwd/bwd so its seeded GD trajectory stays
    numerically identical to index.html / _preview.py."""
    def __init__(self, spec, p, in_shape, oc):
        self.spec, self.p, self.in_shape, self.oc = spec, p, in_shape, oc

    def forward(self, th, X):
        return fwd(th, self.spec, X)[0]

    def vjp(self, th, X, cot):
        out, C = fwd(th, self.spec, X)
        c = cot(out) if callable(cot) else cot
        return bwd(th, self.spec, C, c), out

    def jac_cols(self, th, X):
        out, C = fwd(th, self.spec, X)
        N, oc = out.shape
        M = N * oc
        dO = torch.eye(M, dtype=th.dtype, device=th.device).reshape(M, N, oc)
        return bwd_batch(th, self.spec, C, dO), out.reshape(-1)

    def init_theta(self, seed, initScale):
        return init_theta(self.spec, self.p, initScale, seed, getattr(self, "init_scheme", "default"))


# ----- loss abstraction: MSE keeps every panel valid; CE = softmax classification -----
class MSELoss:
    """Squared error. Handles two shapes (sum over the last/output axis):
      * regression / one-hot classification — out (N, C),    Y (N, C) float       → ÷N
      * next-token LM (OpenWebText)          — out (N, T, V), Y token ids (N, T)   → one-hot, ÷N·T
    so the language model can be trained with MSE (regress the logits toward the one-hot next token).
    The per-token normalisation mirrors CELoss, so the same learning rate sits at a comparable scale
    whether you pick MSE or CE on OWT. MIRRORS eos_lab.models.MSELoss."""
    name = "mse"

    @staticmethod
    def _tokens(out, N):
        return N * out.shape[1] if out.dim() == 3 else N      # per-token (LM) vs per-sample

    @staticmethod
    def _target(out, Y):
        if Y.dtype == torch.long:                             # LM integer targets → one-hot on the fly
            return torch.zeros_like(out).scatter_(-1, Y.unsqueeze(-1), 1.0)
        return Y

    def value(self, out, Y, N):
        return 0.5 * ((out - self._target(out, Y)) ** 2).sum() / self._tokens(out, N)

    def resid_cotangent(self, out, Y, N):       # ∂L/∂out (so ∇_θL = vjp(this))
        return (out - self._target(out, Y)) / self._tokens(out, N)

    def gn_apply(self, out, Jv, N):             # output-space Hessian A applied to Jv (A = I/tokens)
        return Jv / self._tokens(out, N)


class CELoss:
    """Softmax cross-entropy over the last axis. Handles classification (out (N,C), Y one-hot (N,C),
    ÷N) and next-token LM (out (N,T,V), Y token ids long (N,T), ÷N·T per-token). MIRRORS eos_lab."""
    name = "ce"

    @staticmethod
    def _tokens(out, N):
        return N * out.shape[1] if out.dim() == 3 else N

    @staticmethod
    def _onehot(out, Y):
        if Y.dtype == torch.long:
            return torch.zeros_like(out).scatter_(-1, Y.unsqueeze(-1), 1.0)
        return Y

    def value(self, out, Y, N):
        logp = torch.log_softmax(out, dim=-1)
        if Y.dtype == torch.long:
            nll = -logp.gather(-1, Y.unsqueeze(-1)).squeeze(-1)
            return nll.sum() / self._tokens(out, N)
        return -(Y * logp).sum() / self._tokens(out, N)

    def resid_cotangent(self, out, Y, N):
        return (torch.softmax(out, dim=-1) - self._onehot(out, Y)) / self._tokens(out, N)

    def gn_apply(self, out, Jv, N):             # A = (diag(p) − ppᵀ)/tokens
        p = torch.softmax(out, dim=-1)
        return (p * Jv - p * (p * Jv).sum(dim=-1, keepdim=True)) / self._tokens(out, N)


def build_loss(name):
    return CELoss() if name == "ce" else MSELoss()


# ===================== autograd architectures (GPU-server only) =====================
# Parameters live in ONE flat th vector (so all the matrix-free Lanczos/HVP machinery is unchanged);
# forward is built functionally from th slices; grads and EXACT HVPs come from autograd (no
# finite-difference cancellation — important at fp32 / millions of params). No browser counterpart,
# so the seeded init uses torch's RNG (fast, reproducible) rather than mulberry32.
class AutogradModel(Model):
    exact_hvp = True

    def __init__(self):
        self._specs = []      # [name, shape, numel, offset, fan_in, init] ; init ∈ {'n','0','1'}
        self._p = 0

    def _add(self, name, shape, fan_in=None, init="n"):
        numel = 1
        for s in shape:
            numel *= s
        self._specs.append((name, tuple(shape), numel, self._p, fan_in, init))
        self._p += numel

    @property
    def p(self):
        return self._p

    def _unflatten(self, th):
        return {name: th[off:off + numel].view(*shape)
                for name, shape, numel, off, fan_in, init in self._specs}

    def init_theta(self, seed, initScale):
        g = torch.Generator(device="cpu")
        g.manual_seed((int(seed) & 0x7FFFFFFF) or 1)
        scheme = getattr(self, "init_scheme", "default")
        readout_off = max((off for _, _, _, off, _, ini in self._specs if ini == "n"), default=-1)
        parts = []
        for name, shape, numel, off, fan_in, init in self._specs:
            if init == "0":
                parts.append(torch.zeros(numel))
            elif init == "1":
                parts.append(torch.ones(numel))
            else:
                fi = fan_in or numel
                fo = max(1, numel // fi)
                kind, sc = init_kind_scale(scheme, initScale, fi, fo, off == readout_off)
                draw = (torch.randn(numel, generator=g) if kind == "normal"
                        else (torch.rand(numel, generator=g) * 2.0 - 1.0))
                parts.append(draw * sc)
        return torch.cat(parts).to(dtype=DTYPE, device=_dev())

    def forward(self, th, X):
        return self._net(self._unflatten(th), X)

    def vjp(self, th, X, cot):
        with torch.enable_grad():
            thl = th.detach().requires_grad_(True)
            out = self._net(self._unflatten(thl), X)
            c = cot(out) if callable(cot) else cot
            g, = torch.autograd.grad((out * c.detach()).sum(), thl)
        return g, out.detach()

    def hvp(self, th, X, kind, v, Y=None, c=None):
        """Exact Hessian–vector products via double-backward (Pearlmutter)."""
        N = X.shape[0]
        with torch.enable_grad():
            if kind == "G":                                   # Gauss-Newton Jᵀ A J v (loss-aware A)
                fn = lambda t: self._net(self._unflatten(t), X)
                out, Jv = torch.func.jvp(fn, (th,), (v,))
                AJv = _TL.loss.gn_apply(out.detach(), Jv.detach(), N)
                thl = th.detach().requires_grad_(True)
                o2 = self._net(self._unflatten(thl), X)
                Gv, = torch.autograd.grad((o2 * AJv).sum(), thl)
                return Gv.detach()
            thl = th.detach().requires_grad_(True)
            out = self._net(self._unflatten(thl), X)
            if kind == "F":
                scal = out.sum()
            elif kind == "L":
                scal = _TL.loss.value(out, Y, N)
            else:                                             # "S": Σ_a c_a f_a → Hessian = Σ c_a ∇²f_a
                cc = c.detach() if torch.is_tensor(c) else c
                scal = (out * cc).sum()
            g, = torch.autograd.grad(scal, thl, create_graph=True)
            Hv, = torch.autograd.grad((g * v).sum(), thl)
            return Hv.detach()

    def jac_cols(self, th, X):
        """Per-output Jacobian J (M,p) + flat out (M,), computed with a VECTORIZED vjp
        (torch.func.jacrev) instead of an M-long Python backward loop — ~20× faster for the
        transformer/conv nets (M=N·oc backward passes collapse into a few chunked batched passes).
        Chunked to bound vmap memory."""
        thl = th.detach()

        def f(t):
            return self._net(self._unflatten(t), X).reshape(-1)

        out = f(thl).detach()
        M = out.shape[0]
        chunk = max(16, min(M, 8_000_000 // max(self.p, 1)))   # bound the vmapped backward's memory
        with torch.enable_grad():
            try:
                J = torch.func.jacrev(f, chunk_size=chunk)(thl)
            except TypeError:                                  # older torch without chunk_size
                J = torch.func.jacrev(f)(thl)
        return J, out

    def _net(self, p, X):
        raise NotImplementedError


class CnnModel(AutogradModel):
    """Small conv net for CIFAR-10: 3 conv blocks (conv→act→avg-pool) + global-avg-pool + linear head.
    Average (not max) pooling keeps the network smooth so the loss Hessian / sharpness are well defined."""
    def __init__(self, inDim, outDim, P):
        super().__init__()
        assert inDim == 3 * 32 * 32, "CNN expects CIFAR-shaped input (3×32×32 = 3072)"
        self.in_shape, self.oc = (3, 32, 32), outDim
        self.act = P.get("act", "tanh")
        m = max(0.1, float(P.get("chmul", 1.0)))
        cin, self._convs = 3, []
        for j, c in enumerate(max(1, int(round(c * m))) for c in (32, 64, 64)):
            self._add(f"c{j}w", (c, cin, 3, 3), fan_in=cin * 9)
            self._add(f"c{j}b", (c,), init="0")
            self._convs.append((f"c{j}w", f"c{j}b"))
            cin = c
        self._add("fcw", (outDim, cin), fan_in=cin)
        self._add("fcb", (outDim,), init="0")

    def _net(self, p, X):
        h = X.view(X.shape[0], 3, 32, 32)
        for wn, bn in self._convs:
            h = F.avg_pool2d(actf(self.act, F.conv2d(h, p[wn], p[bn], padding=1)), 2)
        return h.mean(dim=(2, 3)) @ p["fcw"].t() + p["fcb"]        # global avg pool → (N, oc)


class Vgg11Model(AutogradModel):
    """VGG11 conv stack (CIFAR-adapted, no BatchNorm, smooth avg-pool) scaled by chmul + a linear head.
    chmul=1 → full ~9.4M-param VGG11 (needs the 10M cap); chmul=0.25 → ~0.6M (fast default).
    Avg (not max) pooling keeps curvature well defined for the sharpness/Hessian analysis."""
    CFG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self, inDim, outDim, P):
        super().__init__()
        assert inDim == 3 * 32 * 32, "VGG11 expects CIFAR-shaped input (3×32×32 = 3072)"
        self.in_shape, self.oc = (3, 32, 32), outDim
        self.act = P.get("act", "tanh")
        m = max(0.05, float(P.get("chmul", 0.25)))
        cin, j, self._layers = 3, 0, []
        for v in self.CFG:
            if v == "M":
                self._layers.append(("M",))
            else:
                c = max(1, int(round(v * m)))
                self._add(f"c{j}w", (c, cin, 3, 3), fan_in=cin * 9)
                self._add(f"c{j}b", (c,), init="0")
                self._layers.append(("C", f"c{j}w", f"c{j}b"))
                cin, j = c, j + 1
        self._add("fcw", (outDim, cin), fan_in=cin)              # 5 maxpools: 32→1, so feature = cin
        self._add("fcb", (outDim,), init="0")

    def _net(self, p, X):
        h = X.view(X.shape[0], 3, 32, 32)
        for L in self._layers:
            h = F.avg_pool2d(h, 2) if L[0] == "M" else actf(self.act, F.conv2d(h, p[L[1]], p[L[2]], padding=1))
        return h.view(h.shape[0], -1) @ p["fcw"].t() + p["fcb"]


class GptModel(AutogradModel):
    """Mini-GPT-style transformer for the sorting task: scalar→d_model embed + learned positional,
    nlayer pre-norm blocks (multi-head self-attention + MLP), linear head → one scalar per position.
    Full (non-causal) self-attention so the sorting target is learnable. oc = sequence length L."""
    def __init__(self, inDim, outDim, P):
        super().__init__()
        self.L, self.oc, self.in_shape = inDim, outDim, (inDim,)
        self.pool = (P.get("dataset") == "ksparse")     # k-sparse parity: mean-pool the per-position head to ONE scalar (oc=1)
        d, h, nl = int(P.get("dmodel", 64)), int(P.get("nhead", 4)), int(P.get("nlayer", 2))
        assert d % h == 0, "dmodel must be divisible by nhead"
        self.d, self.h, self.nl = d, h, nl
        self._add("emb_w", (d, 1), fan_in=1)
        self._add("emb_b", (d,), init="0")
        self._add("pos", (self.L, d), fan_in=d)
        for i in range(nl):
            q = f"b{i}_"
            self._add(q + "ln1g", (d,), init="1"); self._add(q + "ln1b", (d,), init="0")
            self._add(q + "qkvw", (3 * d, d), fan_in=d); self._add(q + "qkvb", (3 * d,), init="0")
            self._add(q + "projw", (d, d), fan_in=d); self._add(q + "projb", (d,), init="0")
            self._add(q + "ln2g", (d,), init="1"); self._add(q + "ln2b", (d,), init="0")
            self._add(q + "fc1w", (4 * d, d), fan_in=d); self._add(q + "fc1b", (4 * d,), init="0")
            self._add(q + "fc2w", (d, 4 * d), fan_in=4 * d); self._add(q + "fc2b", (d,), init="0")
        self._add("lnfg", (d,), init="1"); self._add("lnfb", (d,), init="0")
        self._add("headw", (1, d), fan_in=d); self._add("headb", (1,), init="0")

    @staticmethod
    def _ln(x, g, b):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return (x - mu) / torch.sqrt(var + 1e-5) * g + b

    def _net(self, p, X):
        N, L, d, h = X.shape[0], self.L, self.d, self.h
        dh = d // h
        z = X.view(N, L, 1) @ p["emb_w"].t() + p["emb_b"] + p["pos"].unsqueeze(0)   # (N,L,d)
        for i in range(self.nl):
            q = f"b{i}_"
            a = self._ln(z, p[q + "ln1g"], p[q + "ln1b"])
            qkv = a @ p[q + "qkvw"].t() + p[q + "qkvb"]                              # (N,L,3d)
            qq, kk, vv = qkv.split(d, dim=2)
            qq = qq.view(N, L, h, dh).transpose(1, 2)                               # (N,h,L,dh)
            kk = kk.view(N, L, h, dh).transpose(1, 2)
            vv = vv.view(N, L, h, dh).transpose(1, 2)
            att = torch.softmax((qq @ kk.transpose(-2, -1)) / math.sqrt(dh), dim=-1)
            o = (att @ vv).transpose(1, 2).reshape(N, L, d)
            z = z + (o @ p[q + "projw"].t() + p[q + "projb"])
            a2 = self._ln(z, p[q + "ln2g"], p[q + "ln2b"])
            mm = F.gelu(a2 @ p[q + "fc1w"].t() + p[q + "fc1b"])
            z = z + (mm @ p[q + "fc2w"].t() + p[q + "fc2b"])
        z = self._ln(z, p["lnfg"], p["lnfb"])
        out = z @ p["headw"].t() + p["headb"]                                         # (N, L, 1)
        return out.mean(1) if self.pool else out.view(N, L)                          # ksparse: (N,1) pooled scalar parity · sorting: (N, L)


class GptLMModel(AutogradModel):
    """Mini-GPT language model for OpenWebText: token-embedding lookup + learned positional, nlayer
    pre-norm blocks (CAUSAL self-attention + MLP), final LN, weight-TIED head (logits = h·Wteᵀ).
    Input X is (N, block) int token ids; output (N, block, vocab) next-token logits (cross-entropy).
    MIRRORS eos_lab.models.GptLMModel. oc = vocab."""
    def __init__(self, block, vocab, P):
        super().__init__()
        self.L, self.V, self.oc, self.in_shape = block, vocab, vocab, (block,)
        d, h, nl = int(P.get("dmodel", 64)), int(P.get("nhead", 4)), int(P.get("nlayer", 2))
        assert d % h == 0, "dmodel must be divisible by nhead"
        self.d, self.h, self.nl = d, h, nl
        self._add("wte", (vocab, d), fan_in=d)          # token embedding (also the tied output head)
        self._add("wpe", (block, d), fan_in=d)          # learned positional
        for i in range(nl):
            q = f"b{i}_"
            self._add(q + "ln1g", (d,), init="1"); self._add(q + "ln1b", (d,), init="0")
            self._add(q + "qkvw", (3 * d, d), fan_in=d); self._add(q + "qkvb", (3 * d,), init="0")
            self._add(q + "projw", (d, d), fan_in=d); self._add(q + "projb", (d,), init="0")
            self._add(q + "ln2g", (d,), init="1"); self._add(q + "ln2b", (d,), init="0")
            self._add(q + "fc1w", (4 * d, d), fan_in=d); self._add(q + "fc1b", (4 * d,), init="0")
            self._add(q + "fc2w", (d, 4 * d), fan_in=4 * d); self._add(q + "fc2b", (d,), init="0")
        self._add("lnfg", (d,), init="1"); self._add("lnfb", (d,), init="0")

    @staticmethod
    def _ln(x, g, b):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return (x - mu) / torch.sqrt(var + 1e-5) * g + b

    def _net(self, p, X):
        N, T, d, h = X.shape[0], X.shape[1], self.d, self.h
        dh = d // h
        z = p["wte"][X.long()] + p["wpe"][:T].unsqueeze(0)                           # (N,T,d)
        mask = torch.full((T, T), float("-inf"), device=X.device, dtype=z.dtype).triu(1)   # causal
        for i in range(self.nl):
            q = f"b{i}_"
            a = self._ln(z, p[q + "ln1g"], p[q + "ln1b"])
            qkv = a @ p[q + "qkvw"].t() + p[q + "qkvb"]
            qq, kk, vv = qkv.split(d, dim=2)
            qq = qq.view(N, T, h, dh).transpose(1, 2)
            kk = kk.view(N, T, h, dh).transpose(1, 2)
            vv = vv.view(N, T, h, dh).transpose(1, 2)
            att = torch.softmax((qq @ kk.transpose(-2, -1)) / math.sqrt(dh) + mask, dim=-1)
            o = (att @ vv).transpose(1, 2).reshape(N, T, d)
            z = z + (o @ p[q + "projw"].t() + p[q + "projb"])
            a2 = self._ln(z, p[q + "ln2g"], p[q + "ln2b"])
            mm = F.gelu(a2 @ p[q + "fc1w"].t() + p[q + "fc1b"])
            z = z + (mm @ p[q + "fc2w"].t() + p[q + "fc2b"])
        z = self._ln(z, p["lnfg"], p["lnfb"])
        return z @ p["wte"].t()                                                      # tied head (N,T,vocab)


def hvpF(th, X, v):
    """Function-Hessian (∇²Σf) · v."""
    if _TL.model.exact_hvp:
        return _TL.model.hvp(th, X, "F", v)
    return (gradF(th + EPS * v, X)[0] - gradF(th - EPS * v, X)[0]) / (2 * EPS)


def hvpL(th, X, Y, v):
    """Loss-Hessian (∇²L) · v."""
    if _TL.model.exact_hvp:
        return _TL.model.hvp(th, X, "L", v, Y=Y)
    return (gradL(th + EPS * v, X, Y)[0] - gradL(th - EPS * v, X, Y)[0]) / (2 * EPS)


def hvpS(th, X, v, c):
    """Residual term (Σ_a c_a ∇²f_a) · v."""
    if _TL.model.exact_hvp:
        return _TL.model.hvp(th, X, "S", v, c=c)
    return (gradW(th + EPS * v, X, c) - gradW(th - EPS * v, X, c)) / (2 * EPS)


def hvpG(th, X, v):
    """Gauss-Newton (Jᵀ A J) · v, loss-aware output Hessian A."""
    N = X.shape[0]
    if _TL.model.exact_hvp:
        return _TL.model.hvp(th, X, "G", v)
    fp = _TL.model.forward(th + EPS * v, X)
    fm = _TL.model.forward(th - EPS * v, X)
    f0 = _TL.model.forward(th, X)
    AJv = _TL.loss.gn_apply(f0, (fp - fm) / (2 * EPS), N)
    return gradW(th, X, AJv)


def grad_out(th, X, a):
    """∇f_a — gradient of a single (flattened) output index a."""
    def cot(out):
        N, oc = out.shape
        dO = torch.zeros(N, oc, dtype=out.dtype, device=out.device)
        dO.reshape(-1)[a] = 1.0
        return dO
    return _TL.model.vjp(th, X, cot)[0]


def jac_cols(th, X):
    """Per-output Jacobian J (M,p) (row a = ∇f_a) + flat out (M,)."""
    return _TL.model.jac_cols(th, X)


def jac_hvp(th, X, z):
    """{∇²f_a · z}_a — central difference of jac_cols. Shape (M, p)."""
    cp, _ = jac_cols(th + EPS * z, X)
    cm, _ = jac_cols(th - EPS * z, X)
    return (cp - cm) / (2 * EPS)


def hvpG2(th, X, v):
    """G2 = Σ_a (∇²f_a)² applied to v."""
    u = jac_hvp(th, X, v)
    out = torch.zeros(th.shape[0], dtype=th.dtype, device=th.device)
    for a in range(u.shape[0]):
        out += (grad_out(th + EPS * u[a], X, a)
                - grad_out(th - EPS * u[a], X, a)) / (2 * EPS)
    return out


def pin_sign(v):
    """Deterministic absolute sign for an eigenvector: largest-|component| positive."""
    mi = int(torch.argmax(torch.abs(v)))
    return v if float(v[mi]) >= 0 else -v


def sym_eig_desc(A):
    """Eigen-decomposition of a small symmetric matrix, descending. Returns (vals, vecs cols)."""
    w, V = _safe_eigh(A)
    idx = torch.argsort(w, descending=True)
    return w[idx], V[:, idx]


def fh_frozen(th, X, nProbe, n7, nM, mV, M):
    """§9 frozen function-Hessian-tensor SVD at θ: returns (gamma[i], y[i] R^M, z[i][j] R^p, tau[i][j])."""
    Jc, _ = jac_cols(th, X)
    GH = torch.zeros(M, M, dtype=DTYPE, device=_dev())
    for pr in range(nProbe):
        Hz = jac_hvp(th, X, _randn_vec(Jc.shape[1], (0xC0FFEE + pr * 40503) & 0xFFFFFFFF))
        GH += Hz @ Hz.t()
    GH /= nProbe
    Hv, Vh = sym_eig_desc(GH)                       # right singular vecs y_i = Vh[:,i]
    nn = min(n7, M)
    out = {"gamma": [], "y": [], "z": [], "tau": []}
    for i in range(nn):
        out["gamma"].append(math.sqrt(max(float(Hv[i]), 1e-30)))
        out["y"].append(Vh[:, i])
        tv, bv, tval, bval = lanczos_extreme(
            lambda v: hvpS(th, X, v, Vh[:, i].reshape(X.shape[0], -1)), Jc.shape[1], nM, mV,
            (0x9E37 + i * 0x1000) & 0xFFFFFFFF)
        out["z"].append(list(tv) + list(bv))
        out["tau"].append(list(tval) + list(bval))
    return out


def cubic_init_state():
    """Fresh per-run §10 cubic-window state (mirrors eos_lab Diagnostics' _c* attributes)."""
    return {"T0": -1, "Th0": None, "Base": 0.0, "J0": None, "Z": None, "Q0z": None, "dQz": None,
            "Jp": None, "HistR": None, "HistJ": None, "A47": 0.0, "P47": 0.0,
            "J": None, "HistG": None, "A51": 0.0, "P51": 0.0, "A51n": 0.0, "P51n": 0.0}


def cubic_step(ctx, st, th, X, Y, t, J, out, rr, bEk_vals, shH):
    """§10 CUBIC approximation (paper §6) — Eq-47 (single) / Eq-51 (multi) σ₁ predictions with EXACT J&Q
    propagation. At each window start freeze θ₀, J₀ and the 3rd-derivative tensor T (matrix-free central
    differences of the HVPs); within the window propagate J and Q (Q via the residual·Jacobian history →
    O(window²) HVPs) and accumulate the per-step NTK-σ₁ change. Returns the per-step keys (c47/c47p,
    c51/c51p, c51n/c51np ÷N, cActN, cActH, cdQ, cdJ); mutates `st`. MIRRORS eos_lab Diagnostics._cubic_step."""
    N, p, M = ctx["N"], ctx["p"], ctx["M"]
    lr, ee, cubW, cn = ctx["lr"], ctx["ee"], ctx["cubicapprox"], ctx["cn"]
    multi, multi_ok = ctx["multi"], ctx["multi_ok"]
    etaN = lr / max(N, 1); reps_ = max(1, ee); eps = 1e-2
    dev, dt = _dev(), DTYPE
    rec = {}
    if multi and not multi_ok:
        return rec
    th0 = st["Th0"]

    def T2s(a, b):                                  # single: T[a,b]=∇³f(θ₀)[a,b] (p,) via central diff of Q·b
        an = float(a.norm())
        if an < 1e-30:
            return torch.zeros(p, dtype=dt, device=dev)
        ah = a / an
        return an * (hvpF(th0 + eps * ah, X, b) - hvpF(th0 - eps * ah, X, b)) / (2 * eps)

    def T2m(a, b):                                  # multi: T[a,b]={∇³f_c(θ₀)[a,b]}_c (M,p) via central diff of Q[b]
        an = float(a.norm())
        if an < 1e-30:
            return torch.zeros(M, p, dtype=dt, device=dev)
        ah = a / an
        return an * (jac_hvp(th0 + eps * ah, X, b) - jac_hvp(th0 - eps * ah, X, b)) / (2 * eps)

    if st["T0"] < 0 or (t - st["T0"]) >= cubW:      # window start: freeze θ₀, J₀, probes; reset accumulators
        st["T0"] = t; st["Th0"] = th.clone(); th0 = st["Th0"]
        st["Z"] = [_randn_vec(p, (0xCB1C + i * 0x9E3779B1) & 0xFFFFFFFF) for i in range(cn)]
        if multi:
            J0, _ = jac_cols(th0, X)
            st["J0"] = J0; st["J"] = J0.clone(); st["HistG"] = []
            st["Base"] = float(bEk_vals[0]) if bEk_vals is not None else float(
                _safe_eigvalsh(J0 @ J0.t())[-1])
            st["Q0z"] = [jac_hvp(th0, X, z) for z in st["Z"]]
            st["dQz"] = [torch.zeros(M, p, dtype=dt, device=dev) for _ in st["Z"]]
            st["A51"] = st["P51"] = st["A51n"] = st["P51n"] = 0.0
        else:
            J0 = J
            st["J0"] = J0.clone(); st["Jp"] = J0.clone(); st["HistR"] = []; st["HistJ"] = []
            st["Base"] = float(J0 @ J0)
            st["Q0z"] = [hvpF(th0, X, z) for z in st["Z"]]
            st["dQz"] = [torch.zeros(p, dtype=dt, device=dev) for _ in st["Z"]]
            st["A47"] = st["P47"] = 0.0

    q0n = math.sqrt(max(sum(float((z * z).sum()) for z in st["Q0z"]), 1e-30))
    dqn = math.sqrt(max(sum(float((z * z).sum()) for z in st["dQz"]), 0.0))
    cdQ = 100.0 * dqn / q0n

    if multi and st["J"] is not None:
        actN = float(bEk_vals[0]); cap = 10.0 * max(actN, 1e-30)
        clmp = lambda x: min(max(x, 0.0), cap)
        rec["c51"] = clmp(st["Base"] + st["A51"]) / N
        rec["c51p"] = clmp(st["Base"] + st["A51"] + st["P51"]) / N
        rec["c51n"] = clmp(st["Base"] + st["A51n"]) / N
        rec["c51np"] = clmp(st["Base"] + st["A51n"] + st["P51n"]) / N
        rec["cActN"] = actN / N; rec["cActH"] = shH
        rec["cdJ"] = 100.0 * float((st["J"] - st["J0"]).norm()) / max(float(st["J0"].norm()), 1e-30)
        rec["cdQ"] = cdQ
        for _ in range(reps_):
            Jc = st["J"]; Kc, Vc = sym_eig_desc(Jc @ Jc.t())
            u1 = pin_sign(Vc[:, 0]); sg = math.sqrt(max(float(Kc[0]), 1e-30))
            v1 = Jc.t() @ u1; v1 = v1 / max(float(v1.norm()), 1e-30)
            g = Jc.t() @ rr
            Qg = jac_hvp(th0, X, g)
            for gk in st["HistG"]:
                Qg = Qg + etaN * T2m(gk, g)
            Tgg = T2m(g, g)
            Qgu = (u1.unsqueeze(1) * Qg).sum(0); Tggu = (u1.unsqueeze(1) * Tgg).sum(0)
            dJ = etaN * Qg + 0.5 * (etaN ** 2) * Tgg                        # ΔJ = (η/N)Q_t[g] + ½(η/N)²T[g,g]  (true Taylor ½)
            dJu = dJ.t() @ u1
            st["A51"] += 2 * etaN * sg * float(v1 @ Qgu) + (etaN ** 2) * sg * float(v1 @ Tggu)   # 2√σ₁ v₁ᵀΔJᵀu₁
            st["P51"] += float(dJu @ dJu)
            # "without η²": same cubic trajectory & PSD; drop only the explicit cubic interaction scalar from Δσ₁
            st["A51n"] += 2 * etaN * sg * float(v1 @ Qgu); st["P51n"] += float(dJu @ dJu)
            for i, z in enumerate(st["Z"]):
                st["dQz"][i] = st["dQz"][i] + etaN * T2m(g, z)
            st["HistG"].append(g); st["J"] = Jc + dJ
    elif (not multi) and st["Jp"] is not None:
        actN = float(J @ J); cap = 10.0 * max(actN, 1e-30)
        clmp = lambda x: min(max(x, 0.0), cap)
        rec["c47"] = clmp(st["Base"] + st["A47"]); rec["c47p"] = clmp(st["Base"] + st["A47"] + st["P47"])
        rec["cActN"] = actN; rec["cActH"] = shH
        rec["cdJ"] = 100.0 * float((st["Jp"] - st["J0"]).norm()) / max(float(st["J0"].norm()), 1e-30)
        rec["cdQ"] = cdQ
        rsc = float((Y.reshape(-1) - out.reshape(-1))[0])
        for _ in range(reps_):
            Jc = st["Jp"]; QJ = hvpF(th0, X, Jc)
            for rk, Jk in zip(st["HistR"], st["HistJ"]):
                QJ = QJ + lr * rk * T2s(Jk, Jc)
            TJJ = T2s(Jc, Jc)
            dJ = lr * rsc * QJ + 0.5 * (lr ** 2) * (rsc ** 2) * TJJ        # ΔJ = ηr·Q_t·J + ½η²r²·T[J,J]  (true Taylor ½)
            st["A47"] += 2 * lr * rsc * float(Jc @ QJ) + (lr ** 2) * (rsc ** 2) * float(Jc @ TJJ)   # 2JᵀΔJ
            st["P47"] += float(dJ @ dJ)
            for i, z in enumerate(st["Z"]):
                st["dQz"][i] = st["dQz"][i] + lr * rsc * T2s(Jc, z)
            st["HistR"].append(rsc); st["HistJ"].append(Jc.clone()); st["Jp"] = Jc + dJ
    return rec


def sur_T2(th0, X, a, eps=1e-2):
    """T₀[a,a] = ∇³f(θ₀)[a,a] (M,p) — the frozen 3rd-derivative contracted twice with `a`, via a central
    difference of jac_hvp in the unit `a` direction. Used by the CUBIC surrogate's forward f₀+J₀Δθ+½ΔθᵀQΔθ
    +⅙T[Δθ,Δθ,Δθ] and its Jacobian J₀+Q[Δθ]+½T[Δθ,Δθ]. Returns 0 when ‖a‖≈0 (window start, Δθ=0)."""
    an = float(a.norm())
    base = jac_hvp(th0, X, a)                 # (M,p); ≈0 when a≈0 (also gives the shape for the zero case)
    if an < 1e-30:
        return base * 0.0
    ah = a / an
    return an * (jac_hvp(th0 + eps * ah, X, a) - jac_hvp(th0 - eps * ah, X, a)) / (2 * eps)


# ===================== Lanczos / SLQ =====================
def _randn_vec(p, seed, dt=None):
    g = torch.Generator(device=_dev())
    g.manual_seed(int(seed) & 0x7FFFFFFF)
    return torch.randn(p, dtype=(dt if dt is not None else DTYPE), device=_dev(), generator=g)


def _lanczos_core(hvp, p, m, seed, dt=None):
    dt = dt if dt is not None else DTYPE
    q = _randn_vec(p, seed, dt)
    q = q / q.norm()
    Q = []
    al = []
    be = []
    qp = torch.zeros(p, dtype=dt, device=_dev())
    beta = torch.zeros((), dtype=dt, device=_dev())
    for _ in range(min(p, m)):
        Q.append(q)
        w = hvp(q)
        a = (w @ q)
        al.append(float(a))
        w = w - a * q - beta * qp
        Qm = torch.stack(Q)                       # (k, p) — full reorthogonalization as two matmuls
        for _ in range(2):                          # twice-iterated classical Gram–Schmidt (stable)
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
    mu, Sv = _safe_eigh(T)                   # ascending, on CPU float64 (robust to degenerate tridiagonals: cifar2/mnist2 flat regions)
    idx = torch.argsort(mu, descending=True)
    return mu, Sv, idx


def lanczos_extreme(hvp, p, n, m, seed, dt=None):
    dt = dt if dt is not None else DTYPE
    Q, T, k = _lanczos_core(hvp, p, m, seed, dt)
    mu, Sv, idx = _eig_sorted(T)
    Qmat = torch.stack(Q)                     # (k, p) on _dev()

    def ritz(col):
        c = Sv[:, col].to(device=_dev(), dtype=dt)
        return c @ Qmat                       # (p,)

    topVecs = [ritz(int(idx[min(i, k - 1)])) for i in range(n)]
    botVecs = [ritz(int(idx[max(0, k - 1 - i)])) for i in range(n)]
    topVals = [float(mu[int(idx[min(i, k - 1)])]) for i in range(n)]
    botVals = [float(mu[int(idx[max(0, k - 1 - i)])]) for i in range(n)]
    return topVecs, botVecs, topVals, botVals


def batched_lanczos_extreme(hvp_batch, p, N, n, m, seeds, dt=None):
    """N independent Lanczos eigendecompositions run IN LOCKSTEP (vectorised over the N samples). hvp_batch(V)
    maps (N,p) → (N,p) = {Q_i·V_i}_i (one batched HVP per step instead of N separate ones). Returns the top-n &
    bottom-n eigenpairs per sample: (TV,TW,BV,BW) with shapes (N,n,p),(N,n),(N,n,p),(N,n). Twice-reorthogonalised
    like lanczos_extreme; replacing N×m tiny ops with m batched ops is a large GPU speed-up for §12.
    MIRRORS eos_lab.linalg.batched_lanczos_extreme."""
    dev, dt = _dev(), (dt if dt is not None else DTYPE)
    q = torch.stack([_randn_vec(p, seeds[i], dt) for i in range(N)])  # (N,p)
    q = q / q.norm(dim=1, keepdim=True)
    Qs, al, be = [], [], []
    qp = torch.zeros(N, p, dtype=dt, device=dev); beta = torch.zeros(N, dtype=dt, device=dev)
    done = torch.zeros(N, dtype=torch.bool, device=dev)              # samples whose Krylov space is exhausted
    bmax = torch.zeros(N, dtype=dt, device=dev)                      # running max β (scale, for the relative breakdown test)
    for _ in range(min(p, m)):
        Qs.append(q)
        w = hvp_batch(q)                                            # frozen samples have q=0 ⇒ w=0 ⇒ a,β=0 (stay frozen)
        a = (w * q).sum(1); al.append(a)
        w = w - a[:, None] * q - beta[:, None] * qp
        Qm = torch.stack(Qs, dim=1)                                   # (N,k,p)
        for _ in range(2):                                           # twice-iterated full reorthogonalisation
            w = w - torch.bmm(Qm.transpose(1, 2), torch.bmm(Qm, w.unsqueeze(2))).squeeze(2)
        beta = w.norm(dim=1)
        bmax = torch.maximum(bmax, beta)
        # FREEZE on breakdown (β≈0 ⇒ Krylov space exhausted, e.g. a low-rank Hessian): set q=0 so no noise vector
        # pollutes the basis — the batched equivalent of lanczos_extreme's `if β<tol: break`. Without this the
        # corrupted basis yields non-unit eigenvectors (|cos|>1).
        done = done | (beta < 1e-6 * bmax.clamp_min(1e-30) + 1e-30)
        beta = torch.where(done, torch.zeros_like(beta), beta)       # frozen ⇒ 0 off-diagonal (block-separates T)
        be.append(beta)
        qp = q
        q = torch.where(done.unsqueeze(1), torch.zeros_like(w), w / beta.clamp_min(1e-30).unsqueeze(1))
    Qmat = torch.stack(Qs, dim=1); k = len(Qs)
    T = torch.zeros(N, k, k, dtype=dt, device=dev)
    for i in range(k):
        T[:, i, i] = al[i]
    for i in range(k - 1):
        T[:, i, i + 1] = be[i]; T[:, i + 1, i] = be[i]
    mu, Sv = _safe_eigh(T)                                           # (N,k) ascending, (N,k,k) — robust to degenerate per-sample T
    nn = min(n, k)
    top = mu.argsort(dim=1, descending=True)[:, :nn]; bot = mu.argsort(dim=1)[:, :nn]

    def gp(ix):
        vals = torch.gather(mu, 1, ix)
        Svs = torch.gather(Sv, 2, ix.unsqueeze(1).expand(N, k, nn))   # (N,k,nn)
        vecs = torch.einsum('nkp,nkj->njp', Qmat, Svs)               # (N,nn,p)
        # Real Ritz vectors have ‖·‖≈1 (orthonormal Krylov basis); on breakdown the surplus requested eigenpairs are
        # SPURIOUS null-space vectors with ‖·‖≈0. Make the distinction EXACT: real → unit-normalized; spurious → 0
        # vector AND 0 eigenvalue (a clean sentinel every §12/13/14 consumer excludes; also avoids amplifying noise).
        nrm = vecs.norm(dim=2, keepdim=True)
        real = nrm > 0.5
        vecs = torch.where(real, vecs / nrm.clamp_min(1e-30), torch.zeros_like(vecs))   # real → EXACT unit ; spurious → 0
        vals = torch.where(real.squeeze(2), vals, torch.zeros_like(vals))               # spurious eigenvalue → EXACT 0
        return vecs, vals
    tv, tw = gp(top); bv, bw = gp(bot)
    return tv, tw, bv, bw


def lanczos_extreme_vals(hvp, p, n, m, seed):
    _, T, k = _lanczos_core(hvp, p, m, seed)
    mu, _, idx = _eig_sorted(T)
    top = [float(mu[int(idx[min(i, k - 1)])]) for i in range(n)]
    bot = [float(mu[int(idx[max(0, k - 1 - i)])]) for i in range(n)]
    return top, bot


def _hutch_trace(hvp, p, nprobe, seed):
    """Hutchinson trace estimate: trace(A) ≈ (1/k) Σ zᵀ(A z), z Rademacher ±1 (matrix-free, reuses the HVP)."""
    s = 0.0
    for i in range(nprobe):
        z = torch.sign(_randn_vec(p, (seed + i * 7919) & 0x7FFFFFFF))
        s += float(z @ hvp(z))
    return s / max(1, nprobe)


def slq_density(hvp, p, nprobe, m, ngrid, seed):
    import numpy as np
    TH = []
    W = []
    for pr in range(nprobe):
        _, T, k = _lanczos_core(hvp, p, min(p, m), (seed + pr * 1315423) & 0xFFFFFFFF)
        val, V = _safe_eigh(T)                       # robust to degenerate SLQ tridiagonals (cifar2/mnist2 flat regions)
        val = val.numpy()
        v0 = V[0, :].numpy()
        for i in range(k):
            TH.append(float(val[i]))
            W.append(float(v0[i] ** 2 / nprobe))
    TH = np.asarray(TH)
    W = np.asarray(W)
    lo, hi = float(TH.min()), float(TH.max())
    if not hi > lo:
        hi = lo + 1
        lo -= 1
    pad = 0.05 * (hi - lo) + 1e-9
    lo -= pad
    hi += pad
    sigma = max((hi - lo) / 60, 1e-9)
    inv = 1.0 / (sigma * math.sqrt(2 * math.pi))
    x = np.linspace(lo, hi, ngrid)
    y = np.array([float(np.sum(W * np.exp(-0.5 * ((lam - TH) / sigma) ** 2) * inv)) for lam in x])
    return {"x": x.tolist(), "y": y.tolist()}


def pos_subspace(vals, vecs, frac):
    idx = [i for i in range(len(vals)) if vals[i] > 0]
    E = sum(vals[i] for i in idx)
    if E <= 0:
        return []
    c = 0.0
    sub = []
    for i in idx:
        c += vals[i]
        sub.append(vecs[i])
        if c >= frac * E:
            break
    return sub


def neg_subspace(vals, vecs, frac):
    idx = [i for i in range(len(vals)) if vals[i] < 0]
    E = sum(abs(vals[i]) for i in idx)
    if E <= 0:
        return []
    c = 0.0
    sub = []
    for i in idx:
        c += abs(vals[i])
        sub.append(vecs[i])
        if c >= frac * E:
            break
    return sub


def principal_angles(A, B):
    if len(A) == 0 or len(B) == 0:
        return None
    Am = torch.stack(A)
    Bm = torch.stack(B)
    Mc = (Am @ Bm.t()).double().cpu()         # (d1, d2)
    d1, d2 = Mc.shape
    G = Mc @ Mc.t() if d1 <= d2 else Mc.t() @ Mc
    vals = _safe_eigvalsh(G)                  # ascending (robust to degenerate Gram, e.g. scalar-output cifar2/mnist2)
    ang = []
    for v in vals:
        s = math.sqrt(max(0.0, min(1.0, float(v))))
        ang.append(math.degrees(math.acos(max(-1.0, min(1.0, s)))))
    ang.sort()
    return {"mx": ang[-1], "mn": sum(ang) / len(ang)}


def sign_to(cur, prev):
    if prev is None:
        return cur
    return -cur if float(cur @ prev) < 0 else cur


# ===================== datasets =====================
_CIFAR_CACHE = {}


def _find_cifar_dir():
    cands = [CIFAR_DIR,
             os.path.join(DIR, "data", "cifar-10-batches-py"),   # in-repo copy on shared NAS
             os.path.join(DIR, "cifar-10-batches-py"),
             os.path.expanduser("~/.torch/cifar-10-batches-py"),
             os.path.expanduser("~/data/cifar-10-batches-py")]
    for c in cands:
        if c and os.path.isdir(c) and os.path.isfile(os.path.join(c, "data_batch_1")):
            return c
    return None


def _load_cifar_raw():
    """Load all 5 CIFAR-10 train batches, normalized per-channel, as (50000, 3072) + int labels.
    Pure pickle+numpy (no torchvision). Cached per directory."""
    import pickle
    import numpy as np
    d = _find_cifar_dir()
    if d is None:
        raise FileNotFoundError(
            "CIFAR-10 raw batches not found — pass --cifar-dir <dir containing data_batch_1..5>")
    if d in _CIFAR_CACHE:
        return _CIFAR_CACHE[d]
    xs, ys = [], []
    for i in range(1, 6):
        with open(os.path.join(d, f"data_batch_{i}"), "rb") as f:
            b = pickle.load(f, encoding="bytes")
        xs.append(np.asarray(b[b"data"], dtype=np.float32))
        ys.extend(list(b[b"labels"]))
    X = np.concatenate(xs, 0).reshape(-1, 3, 32, 32) / 255.0       # (50000,3,32,32) in [0,1]
    mean = np.array([0.4914, 0.4822, 0.4465], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], np.float32).reshape(1, 3, 1, 1)
    X = ((X - mean) / std).reshape(-1, 3072)                       # channel-major flatten (C,H,W)
    Y = np.asarray(ys, dtype=np.int64)
    _CIFAR_CACHE[d] = (X, Y)
    return X, Y


def _load_cifar_test_raw():
    """The 10000-image CIFAR-10 test_batch, normalised like the train batches. (X (10000,3072), int labels)."""
    import pickle
    import numpy as np
    d = _find_cifar_dir()
    if d is None or not os.path.isfile(os.path.join(d, "test_batch")):
        return None
    key = d + "::test"
    if key in _CIFAR_CACHE:
        return _CIFAR_CACHE[key]
    with open(os.path.join(d, "test_batch"), "rb") as f:
        b = pickle.load(f, encoding="bytes")
    X = np.asarray(b[b"data"], dtype=np.float32).reshape(-1, 3, 32, 32) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], np.float32).reshape(1, 3, 1, 1)
    X = ((X - mean) / std).reshape(-1, 3072)
    Y = np.asarray(list(b[b"labels"]), dtype=np.int64)
    _CIFAR_CACHE[key] = (X, Y)
    return X, Y


def load_cifar(n, seed):
    """A fixed (seeded) subset of n CIFAR-10 images. X (n,3072); Y (n,10) one-hot (MSE & CE both use it)."""
    import numpy as np
    X, Y = _load_cifar_raw()
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    idx = rng.permutation(X.shape[0])[:n]
    Xt = torch.tensor(X[idx], dtype=DTYPE, device=_dev())
    Yt = torch.zeros(n, 10, dtype=DTYPE, device=_dev())
    Yt[torch.arange(n), torch.tensor(Y[idx], dtype=torch.long, device=_dev())] = 1.0
    return Xt, Yt


def load_cifar2(n, seed, ca=0, cb=1):
    """2-class SCALAR-output CIFAR-10: keep only classes {ca,cb}, label class ca → +1 and class cb → −1
    (a single scalar regression target, d_out=1). X (n,3072); Y (n,1). A fixed (seeded) BALANCED subset:
    the first n//2 images of class ca and the first n−n//2 of class cb, then a seeded shuffle. Lets the
    §16/§17 per-sample-Hessian sections run on REAL images (M=N·1=N is small)."""
    import numpy as np
    X, Y = _load_cifar_raw()
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    ia = np.where(Y == int(ca))[0]; ib = np.where(Y == int(cb))[0]
    rng.shuffle(ia); rng.shuffle(ib)
    na = n // 2; nb = n - na
    idx = np.concatenate([ia[:na], ib[:nb]])
    lab = np.concatenate([np.ones(min(na, len(ia)), np.float32), -np.ones(min(nb, len(ib)), np.float32)])
    perm = rng.permutation(len(idx)); idx = idx[perm]; lab = lab[perm]
    Xt = torch.tensor(X[idx], dtype=DTYPE, device=_dev())
    Yt = torch.tensor(lab, dtype=DTYPE, device=_dev()).reshape(-1, 1)
    return Xt, Yt


_MNIST_CACHE = {}


def _find_mnist_dir():
    cands = [os.path.join(DIR, "data", "mnist"), CIFAR_DIR and os.path.join(os.path.dirname(CIFAR_DIR), "mnist"),
             os.path.expanduser("~/data/mnist"), os.path.expanduser("~/.torch/mnist")]
    for c in cands:
        if c and os.path.isdir(c) and os.path.isfile(os.path.join(c, "train-images-idx3-ubyte")):
            return c
    return None


def _load_mnist_raw(split="train"):
    """MNIST as (N,3072): each 28×28 digit zero-padded to 32×32 and replicated to 3 channels, so it is a
    DROP-IN for the cifar-shaped MLP/CNN/VGG (all expect 3×32×32; VGG's 5 max-pools also need ≥32). Per-channel
    MNIST-normalised. Returns (X float (N,3072), Y int labels). Pure numpy (no torchvision). Cached per dir/split."""
    import numpy as np
    d = _find_mnist_dir()
    if d is None:
        raise FileNotFoundError("MNIST raw idx files not found — expected data/mnist/{train,t10k}-images/labels-idx*-ubyte")
    key = d + "::" + split
    if key in _MNIST_CACHE:
        return _MNIST_CACHE[key]
    pre = "train" if split == "train" else "t10k"
    with open(os.path.join(d, f"{pre}-images-idx3-ubyte"), "rb") as f:
        f.read(16); img = np.frombuffer(f.read(), dtype=np.uint8)
    with open(os.path.join(d, f"{pre}-labels-idx1-ubyte"), "rb") as f:
        f.read(8); lab = np.frombuffer(f.read(), dtype=np.uint8)
    n = lab.shape[0]
    img = img.reshape(n, 28, 28).astype(np.float32) / 255.0
    pad = np.zeros((n, 32, 32), np.float32); pad[:, 2:30, 2:30] = img        # centre the 28×28 digit in 32×32
    x = np.repeat(pad[:, None, :, :], 3, axis=1)                             # (n,3,32,32) replicate to 3 channels
    x = (x - 0.1307) / 0.3081                                                # MNIST normalisation (same on all 3 channels)
    X = x.reshape(n, 3072); Y = lab.astype(np.int64)
    _MNIST_CACHE[key] = (X, Y)
    return X, Y


def load_mnist(n, seed):
    """A fixed (seeded) subset of n MNIST images. X (n,3072) [3×32×32 padded]; Y (n,10) one-hot. MIRRORS load_cifar."""
    import numpy as np
    X, Y = _load_mnist_raw("train")
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    idx = rng.permutation(X.shape[0])[:n]
    Xt = torch.tensor(X[idx], dtype=DTYPE, device=_dev())
    Yt = torch.zeros(n, 10, dtype=DTYPE, device=_dev())
    Yt[torch.arange(n), torch.tensor(Y[idx], dtype=torch.long, device=_dev())] = 1.0
    return Xt, Yt


def load_mnist2(n, seed, ca=0, cb=1):
    """2-class SCALAR-output MNIST: classes {ca,cb} → ±1 (d_out=1). X (n,3072); Y (n,1). Balanced seeded
    subset. MIRRORS load_cifar2 (lets §16/§17 per-sample-Hessian sections run on REAL images, M=N small)."""
    import numpy as np
    X, Y = _load_mnist_raw("train")
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    ia = np.where(Y == int(ca))[0]; ib = np.where(Y == int(cb))[0]
    rng.shuffle(ia); rng.shuffle(ib)
    na = n // 2; nb = n - na
    idx = np.concatenate([ia[:na], ib[:nb]])
    lab = np.concatenate([np.ones(min(na, len(ia)), np.float32), -np.ones(min(nb, len(ib)), np.float32)])
    perm = rng.permutation(len(idx)); idx = idx[perm]; lab = lab[perm]
    Xt = torch.tensor(X[idx], dtype=DTYPE, device=_dev())
    Yt = torch.tensor(lab, dtype=DTYPE, device=_dev()).reshape(-1, 1)
    return Xt, Yt


def load_sort(n, L, drng):
    """Sorting task: input a random length-L sequence, target = the ascending-sorted sequence.
    MSE regression, oc = L. Uses the seeded mulberry32 RNG so runs are reproducible."""
    Xl = [[gauss(drng) for _ in range(L)] for _ in range(n)]
    X = torch.tensor(Xl, dtype=DTYPE, device=_dev())
    Y, _ = torch.sort(X, dim=1)
    return X, Y


def load_chebyshev(n, k):
    """Chebyshev regression (Cohen et al. EoS toy task): `n` points evenly spaced on [-1,1] (input, n×1),
    labeled noiselessly by the Chebyshev polynomial T_k of degree `k` (output, n×1). MSE regression.
    T_k via the recurrence T₀=1, T₁=x, Tₘ=2x·Tₘ₋₁−Tₘ₋₂. MIRRORS eos_lab.data.load_chebyshev."""
    x = torch.linspace(-1.0, 1.0, max(int(n), 1), dtype=DTYPE, device=_dev())
    if int(k) <= 0:
        y = torch.ones_like(x)
    else:
        tm2, tm1 = torch.ones_like(x), x.clone()
        for _ in range(2, int(k) + 1):
            tm2, tm1 = tm1, 2 * x * tm1 - tm2
        y = tm1
    return x.unsqueeze(1), y.unsqueeze(1)


def load_ksparse(n, nbits, k, seed):
    """k-sparse parity: input is an `nbits`-dim ±1 bit vector; the target is the PARITY (product) of a
    FIXED random size-k subset S of the bits → scalar ±1 (+1 for even #(−1) over S, −1 for odd). MSE
    regression, oc = 1. The subset S and the bits are drawn from the shared mulberry32 RNG so the
    dataset is byte-identical across server / eos_lab / browser. For the transformer (gpt) the SAME
    nbits-length ±1 vector is read as a length-nbits sequence of bit-tokens and the per-position head
    is mean-pooled to the single scalar parity. MIRRORS eos_lab.data.load_ksparse."""
    nb = max(1, int(nbits)); kk = max(1, min(int(k), nb))
    perm = _shuffle16(nb, (int(seed) ^ 0x4B5A11) & 0x7FFFFFFF)           # fixed sparse subset = first kk of a seeded shuffle of [0,nb)
    S = set(perm[:kk])
    rng = mulberry32(u32(int(seed) * 7919 + 1))                          # same seed family as synthetic; bits drawn row-major
    X = torch.empty(int(n), nb, dtype=DTYPE, device=_dev())
    Y = torch.empty(int(n), 1, dtype=DTYPE, device=_dev())
    for i in range(int(n)):
        prod = 1.0
        for j in range(nb):
            b = 1.0 if rng() < 0.5 else -1.0
            X[i, j] = b
            if j in S:
                prod *= b
        Y[i, 0] = prod
    return X, Y


_OWT_CACHE = {}


def _owt_tokens(split):
    """Memory-mapped uint16 OWT token array. MIRRORS eos_lab.data._owt_tokens."""
    import numpy as np
    for d in [os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "owt"),
              os.path.expanduser("~/data/owt")]:
        if os.path.isfile(os.path.join(d, "train.bin")):
            key = (d, split)
            if key not in _OWT_CACHE:
                _OWT_CACHE[key] = np.memmap(os.path.join(d, f"{split}.bin"), dtype=np.uint16, mode="r")
            return _OWT_CACHE[key]
    raise FileNotFoundError("OpenWebText tokens not found — run `python -m eos_lab.owt_prepare` "
                            "to build data/owt/{train,val}.bin.")


def load_owt(n, block, seed, split="train"):
    """`n` length-`block` token sequences at random offsets; (X, Y) int64 (n, block) with Y the
    next-token shift. MIRRORS eos_lab.data.load_owt. (Token ids are dtype-independent.)"""
    import numpy as np
    tok = _owt_tokens(split)
    hi = len(tok) - block - 1
    if hi <= 0:
        raise ValueError(f"OWT {split}.bin has {len(tok)} tokens; need > block+1 = {block + 1}.")
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    offs = rng.randint(0, hi, size=n)
    X = np.stack([np.asarray(tok[o:o + block], dtype=np.int64) for o in offs])
    Y = np.stack([np.asarray(tok[o + 1:o + 1 + block], dtype=np.int64) for o in offs])
    return (torch.tensor(X, dtype=torch.long, device=_dev()),
            torch.tensor(Y, dtype=torch.long, device=_dev()))


# ===================== model factory =====================
def build_model(arch, inDim, outDim, P):
    if arch == "mlp":
        spec, p = build_spec(inDim, P["width"], P["depth"], P["act"], P["bias"] == "1", outDim)
        m = MlpModel(spec, p, (inDim,), outDim)
    elif arch == "cnn":
        m = CnnModel(inDim, outDim, P)
    elif arch == "vgg11":
        m = Vgg11Model(inDim, outDim, P)
    elif arch == "gpt":
        m = GptLMModel(inDim, outDim, P) if P.get("dataset") == "owt" else GptModel(inDim, outDim, P)
    else:
        raise ValueError(f"unknown arch '{arch}'")
    m.init_scheme = P.get("initscheme", "default")     # weight-init scheme (default/mup/xavier_*/custom)
    return m


# ===================== data + init (shared by both run modes) =====================
def init_data_theta(P, dataset, N, inD, outD):
    """θ (seeded init) + data (X,Y) + §4d sign groups. Synthetic uses the exact mulberry32 RNG so the
    MLP trajectory matches the browser; cifar/sorting load real data."""
    th = _TL.model.init_theta(P["seed"] + 1, P["init"])            # θ first (residual signs need it)
    drng = mulberry32(u32(P["seed"] * 7919 + 1))
    ssign = P.get("ssign", "off")
    tgt = float(P["tgt"])
    inStd = float(P["inputstd"])
    fixedx = P["fixedx"] == "1"
    if dataset == "owt":
        # inD = block size; token-id X/Y. CE has no residual geometry → §4d sign-groups are empty.
        X, Y = load_owt(N, inD, P["seed"])
        empt = torch.empty(0, dtype=torch.long, device=_dev())
        return th, X, Y, empt, empt
    if dataset == "cifar10":
        X, Y = load_cifar(N, P["seed"])
    elif dataset == "cifar2":
        X, Y = load_cifar2(N, P["seed"], P.get("c2a", 0), P.get("c2b", 1))
    elif dataset == "mnist":
        X, Y = load_mnist(N, P["seed"])
    elif dataset == "mnist2":
        X, Y = load_mnist2(N, P["seed"], P.get("c2a", 0), P.get("c2b", 1))
    elif dataset == "sorting":
        X, Y = load_sort(N, inD, drng)
    elif dataset == "chebyshev":
        X, Y = load_chebyshev(N, P.get("degree", 3))
    elif dataset == "ksparse":
        X, Y = load_ksparse(N, inD, P.get("ksparse", 3), P["seed"])     # n ±1 bits → scalar ±1 parity of a fixed k-subset
    elif dataset == "const":
        # iid Gaussian inputs (like synthetic); every target is the CONSTANT POSITIVE |tgt| PLUS optional Gaussian
        # noise of VARIANCE `cvar` (default 0 ⇒ exactly constant). cvar=0 ⇒ all initial residuals r=y−f(x,θ₀) share
        # the SAME (positive) sign (uniform residuals — minimal cross-sample structure); raising cvar spreads the
        # targets ⇒ the residuals decorrelate, dialling up the degree of interaction between samples (§13/§14/§12b).
        cstd = max(0.0, float(P.get("cvar", 0.0))) ** 0.5                  # noise std = √variance
        Xl = [[inStd * gauss(drng) for _ in range(inD)] for _ in range(N)]
        X = torch.tensor(Xl, dtype=DTYPE, device=_dev())
        Ymag = torch.tensor([[abs(tgt) + cstd * gauss(drng) for _ in range(outD)] for _ in range(N)],
                            dtype=DTYPE, device=_dev())                    # |tgt| (+cvar noise); drawn for EVERY ssign (RNG parity)
        if ssign in ("pos", "neg"):                                       # FORCE all initial residual signs: r=y−f=s·max(|tgt|+noise,floor) ⇒ sign s
            s = 1.0 if ssign == "pos" else -1.0
            floor = 0.25 * max(abs(tgt), 1e-6)
            Y = _TL.model.forward(th, X) + s * Ymag.clamp(min=floor)      # uniform |tgt| residual with the chosen sign (cvar still spreads it)
        else:
            Y = Ymag                                                      # off: constant positive target |tgt| (+noise); residuals naturally +
    elif fixedx:
        X = torch.ones(N, inD, dtype=DTYPE, device=_dev())
        if ssign == "off":
            Y = torch.full((N, outD), tgt, dtype=DTYPE, device=_dev())
        else:                                                  # construct Y so r=y−f has a fixed sign, |r|≥floor
            s = 1.0 if ssign == "pos" else -1.0
            floor = 0.25 * max(abs(tgt), 1e-6)
            F = _TL.model.forward(th, X)
            Y = F + s * torch.clamp(torch.full_like(F, abs(tgt)), min=floor)
    elif ssign == "off":
        Xl = [[inStd * gauss(drng) for _ in range(inD)] for _ in range(N)]
        Yl = [[tgt * gauss(drng) for _ in range(outD)] for _ in range(N)]
        X = torch.tensor(Xl, dtype=DTYPE, device=_dev())
        Y = torch.tensor(Yl, dtype=DTYPE, device=_dev())
    else:
        # sample a pool, keep N whose initial residual r=y−f(x,θ₀) is sign-definite with largest |r|
        s = 1.0 if ssign == "pos" else -1.0
        floor = 0.25 * max(abs(tgt), 1e-6)
        K = max(64 * N, 3000)
        Xc = torch.tensor([[inStd * gauss(drng) for _ in range(inD)] for _ in range(K)], dtype=DTYPE, device=_dev())
        Yc = torch.tensor([[tgt * gauss(drng) for _ in range(outD)] for _ in range(K)], dtype=DTYPE, device=_dev())
        Fc = _TL.model.forward(th, Xc)
        Rc = Yc - Fc
        okSign = (Rc * s > 0).all(dim=1)
        mn = Rc.abs().amin(dim=1)
        order = torch.argsort(mn, descending=True).tolist()
        chosen = [k for k in order if bool(okSign[k])]         # genuine same-sign first
        for k in order:                                        # top-up (sign forced below)
            if len(chosen) >= N:
                break
            if not bool(okSign[k]):
                chosen.append(k)
        chosen = chosen[:N]
        X = torch.empty(N, inD, dtype=DTYPE, device=_dev())
        Y = torch.empty(N, outD, dtype=DTYPE, device=_dev())
        for i, k in enumerate(chosen):
            X[i] = Xc[k]
            f = Fc[k]
            Y[i] = f + s * torch.clamp((Yc[k] - f).abs(), min=floor)
    # Fixed-target datasets (cifar10/sorting/chebyshev) load Y directly and so skip the residual-sign
    # construction above — force the requested initial residual sign here by overriding Y per sample
    # (keep the dataset's inputs X and the residual's natural magnitude; only its sign is pinned).
    if dataset in ("cifar10", "cifar2", "mnist", "mnist2", "sorting", "chebyshev", "ksparse") and ssign in ("pos", "neg"):
        s = 1.0 if ssign == "pos" else -1.0
        floor = 0.25 * max(abs(tgt), 1e-6)
        F0 = _TL.model.forward(th, X)
        Y = F0 + s * torch.clamp((Y - F0).abs(), min=floor)
    # §4d: fix the two sample groups by the sign of the summed initial residual r=y−f(x,θ₀)
    ssum0 = (Y - _TL.model.forward(th, X)).sum(dim=1)
    pos_rows = torch.tensor([i * outD + c for i in range(N) if float(ssum0[i]) >= 0 for c in range(outD)],
                            dtype=torch.long, device=_dev())
    neg_rows = torch.tensor([i * outD + c for i in range(N) if float(ssum0[i]) < 0 for c in range(outD)],
                            dtype=torch.long, device=_dev())
    return th, X, Y, pos_rows, neg_rows


# ===================== streaming run =====================
def run_stream(P):
    """Yield SSE message dicts: one 'meta', then ('step' [+ 'slq']) per eig-tick, then 'done'."""
    dataset = P.get("dataset", "synthetic")
    arch = P.get("arch", "mlp")
    if dataset == "cifar10":
        inDimE, outDimE = 3072, 10
    elif dataset == "cifar2":
        inDimE, outDimE = 3072, 1                             # 2-class CIFAR cast as SCALAR regression (±1)
    elif dataset == "mnist":
        inDimE, outDimE = 3072, 10                            # MNIST padded to 3×32×32 (drop-in for cifar-shaped MLP/CNN/VGG)
    elif dataset == "mnist2":
        inDimE, outDimE = 3072, 1                             # 2-class MNIST cast as SCALAR regression (±1)
    elif dataset == "sorting":
        inDimE = outDimE = int(P["seqlen"])
    elif dataset == "owt":
        inDimE, outDimE = int(P["seqlen"]), int(P["vocab"])   # block size, vocab
    elif dataset == "chebyshev":
        inDimE = outDimE = 1                                  # scalar x → scalar T_k(x)
    elif dataset == "ksparse":
        inDimE, outDimE = int(P["indim"]), 1                  # n ±1 bits in → scalar ±1 parity out (gpt: read as a length-n bit sequence, mean-pooled)
    else:
        inDimE, outDimE = P["indim"], P["outdim"]
    _TL.model = build_model(arch, inDimE, outDimE, P)
    _TL.loss = build_loss(P.get("loss", "mse"))
    p = _TL.model.p
    half = max(1, p // 2)
    n = min(max(1, P["neig"]), half)
    kth = min(max(1, P["kth"]), half)
    s1, s2, s3, s4, s5, s6, gs = (P["s1"], P["s2"], P["s3"], P["s4"],
                                  P["s5"], P["s6"], P["gs"])
    s7, s8, s9, s10, s11, s12 = P["s7"], P["s8"], P["s9"], P["s10"], P["s11"], P["s12"]
    s13 = P.get("s13", 0)                # §7a NTK alignment (residual→NTK, NTK→FH-SVD)
    s14 = P.get("s14", 0)                # §9c: σ₁ predictions vs the FULL loss-Hessian sharpness λmax(∇²L)
    s15 = P.get("s15", 0)                # §9d: predictions with the residual self-computed by the quadratic model
    s16 = P.get("s16", 0)                # §9d-c: §9d predictions vs the full loss-Hessian sharpness
    s17 = P.get("s17", 0)                # §10: CUBIC approximation (Eq-47/51 σ₁ predictions, exact J&Q propagation)
    s18 = P.get("s18", 0)                # §11: 3D grids T1=JᵢᵀQⱼJₖ, T2=uⱼuₖT1, T3=rᵢuⱼuₖT1 (multi-sample, small M)
    s19 = P.get("s19", 0)                # §12: per-sample Q_i eigenvector cross-similarity grids/cuboids (small N)
    s20 = P.get("s20", 0)                # §14: Tr(ΔNTK) per-triplet (i,j,k) decomposition cubes (small N; shares §12 eigenpairs)
    s21 = P.get("s21", 0)                # §13: residual-weighted curvature G1/G2/G3 vs exact J_iᵀQ_jJ_k·r_k (own toggle; shares §12 Lanczos + adds N HVPs)
    s22 = P.get("s22", 0)                # §12b: per-sample projection panels (s19=§12a angles+diag-align; both share the §12 Lanczos)
    s23 = P.get("s23", 0)                # §15: 2nd-difference decomposition of ‖J‖²_F & σ₁ (own toggle; holds 3 eig-ticks; multi-only, small-N)
    grid3dcap = max(1, P.get("grid3dcap", 30))
    sec12ncap = max(1, P.get("sec12ncap", 500))    # §12 sample cap (gates N). NOTE: §12 builds dense p×p Hessians
                                                    # (p HVPs + O(p³) eig per sample) → large p is SLOW; keep models small.
    if _TL.loss.name == "ce":            # CE: §7/§7a/§8/§4b/§4c use the function NTK (Jᵤ·Jᵤᵀ) and the generic
        s11 = s12 = s14 = s15 = s16 = s17 = False   # residual r=−∂L/∂z (= softmax−onehot), so they're valid. Off for CE:
                                         #   §4d (sign-groups need a scalar residual) and the whole §9 family
                                         #   (§9/§9b/§9c/§9d/§9d-c — the σ₁ recursion is derived for squared loss only).
    nSub = min(max(n, kth, (10 if s6 else 1)), half)
    # minibatch-for-everything: each step samples `bs` of the `Nfull`-sequence pool and uses that sample
    # for BOTH the GD step and that step's diagnostics. bs==Nfull ⇒ deterministic full batch (default).
    Nfull, inD, outD = P["nsamp"], inDimE, outDimE
    bs = Nfull if int(P.get("batch", 0)) <= 0 else min(int(P["batch"]), Nfull)
    full_batch = bs >= Nfull
    N = bs                                  # diagnostics + loss normalisation run on the minibatch
    M = N * outD
    multi = M > 1
    # The multi-sample sections form the explicit M×p Jacobian Jc (and an M×M Gram), so gate them by a
    # size budget (mirrors eos_lab): feasible for e.g. CIFAR-CE (M≈N·10), infeasible for OWT (M=N·T·vocab).
    multi_ok = multi and M <= 2048 and M * p <= 700_000_000
    # §12a/§12b are also well-defined for a SINGLE sample (N=1): the per-sample projection panels need only one
    # Q_i=∇²f_i. (The pair-based principal-angle / cross-projection panels just stay empty — no i≠j pairs.) §13/§14
    # still require multi (they build N×N×N cubes). Lets you sanity-check §12b against §4 at N=1, where they coincide.
    s12single = (s19 or s22) and not multi and M * p <= 700_000_000
    n7 = min(n, max(M, 1))
    n8 = min(n, max(1, p // 2))
    nResid = M                       # §1 residual plot: ALL samples' residual evolution (was capped at 12)
    lr = P["lr"]
    opt = P.get("optimizer", "gd")
    thr = 2.0 / lr

    if p > PMAX:
        yield {"type": "meta", "error": f"p={p} parameters exceeds the cap ({PMAX}). Reduce width/depth."}
        return
    ce = _TL.loss.name == "ce"
    yield {"type": "meta", "p": p, "n": n, "kth": kth, "nResid": nResid, "thr": thr,
           "n7": n7, "n8": n8, "multi": multi, "multi_ok": multi_ok, "loss": _TL.loss.name, "arch": arch, "dataset": dataset,
           "device": f"{_dev()} · {str(DTYPE).replace('torch.', '')}"}

    # ---- data + init (shared helper): synthetic = mulberry32 (browser-identical); cifar/sorting = real data ----
    th, X, Y, pos_rows, neg_rows = init_data_theta(P, dataset, Nfull, inD, outD)
    Xpool, Ypool, poolPos, poolNeg = X, Y, pos_rows, neg_rows   # full pool; minibatch sampled per-step below

    mF = min(p, max(2 * nSub + 24, 44))
    mV = min(p, max(2 * n + 16, 28))
    mSLQ = min(p, 40)
    steps = P["steps"]
    ee = P["eigevery"]
    nProbe = max(1, P["slqprobes"])
    efrac = min(1.0, max(0.5, P["energyp"] / 100.0))
    slqStride = max(1, math.ceil((steps // ee + 1) / 50))
    heavyevery = max(1, P.get("heavyevery", 4))   # §7-proj + §8 compute every heavyevery-th tick (responsiveness)
    # §11-§14 cube/grid cadence: emit on every `cubeevery`-th diagnostic tick. Heavy at large M (M HVPs + M³ work
    #   each emission) and the N³ cubes dominate the capture file size, so raise `cubeevery` for long all-sections
    #   captures (§15 / §1-§9 / §16-§17 stay full-resolution). 1 = every tick (default, unchanged behaviour).
    g3dstride = max(1, int(P.get("cubeevery", 1)))
    start = max(0, min(int(P.get("start", 0)), steps))  # resume: fast-forward GD to here, then stream

    mytok = P.get("_token", 0)                          # per-device token claimed in _sse via acquire_device()
    _devkey = str(_dev())                               # a newer /run on THIS device bumps RUN_TOKEN[_devkey] → we stop

    t0 = time.time()
    eigTick = 0
    done = 0
    sec14_rhist = []          # §14: per-sample residual history (last ≤5 eig-ticks) for the eq-③ multi-step ratio
    sec14_prev = None         # §14: previous eig-tick's {TV,TW,BV,BW,r,Jg} (u,σ,r_j,J_k at t; cur tick gives J_i,r_k at t+1)
    sec13_prev = None         # §13: previous eig-tick's per-sample {TV,TW,BV,BW,r,Jg} (Q_i,Q_k & J',r at t-1)
    sec13_qjprev = None        # §13 panel 4: previous eig-tick's {QJ=(N_j,N_i,p) Q_iJ_j, r} for the ‖A−B‖/‖A‖ asymmetry
    sec15_hist = []            # §15: rolling buffer of the last 2 eig-ticks' {th, J, r} (need t−1 and t−2)
    sec15_th64 = None          # §15: float64 SHADOW GD trajectory (MLP only). D²=‖J_t‖²−2‖J_{t-1}‖²+‖J_{t-2}‖² is a ~9-digit
                               #   catastrophic cancellation when the net is near-stationary (e.g. chebyshev init=0.2: ‖J‖²≈21,
                               #   true D²≈1e-8). The displayed trajectory is float32 (≈7 digits) ⇒ D² is pure roundoff ⇒ divergence%
                               #   garbage. We mirror the GD in float64 (seeded at θ₀, same per-step minibatch) so the consecutive θ's
                               #   are float64-consistent and D² is clean — matching the float64 browser/eos_lab backends. Verified:
                               #   float64 trajectory ⇒ divP→0.00% exactly (theory IS correct); float32 ⇒ ±500% swings.
    prevTop = [None] * nSub
    prevBot = [None] * nSub
    prevPos = None
    prevNeg = None
    prevNtk = [None] * n7
    prevUJ = [None] * n7
    prevVH = [None] * n7
    prevSharp7a = None; prevNtkR7a = None; prevPrevNtkR7a = None   # §7a Δλ·Δr running products
    sumGsync = [0.0] * n7; sumGlag = [0.0] * n7; cntGs7a = 0; cntGl7a = 0
    prevM1 = [None] * 4
    prevM2 = [None] * 4
    prevG2 = [None] * n8
    prevQ9t = [None] * n
    prevQ9b = [None] * n
    qapprox = max(1, P["qapprox"]); qmode = P["qmode"]; tset = P["tset"]   # §9 window + (legacy qmode) + Eq-29 |T|
    thT0 = -1; thTh0 = None; thBase = 0.0; thJp = None; thJ = None; thFroz = None
    thAcc1 = 0.0; thAcc2 = 0.0; thProd3 = 1.0; thProd4 = 1.0; thProd5 = 1.0
    thAccPSD = 0.0; thAccPSD1 = 0.0     # §9b: accumulated 2nd-order PSD term ‖ΔJᵀu₁‖² (≥0), single/multi
    # §9d: identical predictions, residual SELF-COMPUTED by the frozen quadratic model (closed loop)
    thJ0 = None; thF0 = None; thDth = None; thJ_d = None
    thProd3_d = 1.0; thProd4_d = 1.0; thProd5_d = 1.0; thAcc2_d = 0.0; thAccPSD_d = 0.0
    thJp0 = None; thF0s = 0.0; thDth_s = None; thJp_d = None; thAcc1_d = 0.0; thAccPSD1_d = 0.0
    # §10 cubic-approximation window state + context (see cubic_step)
    cubCtx = {"N": N, "p": p, "M": M, "lr": lr, "ee": ee, "cubicapprox": max(1, P.get("cubicapprox", 10)),
              "cn": max(1, min(nProbe, 4)), "multi": multi, "multi_ok": multi_ok}
    cubSt = cubic_init_state()

    # ── §16: standalone curvature-aligned optimizer (own iteration axis i). Runs from θ₀ in float64, streams g16 records,
    #    then the normal GD loop below proceeds. Independent of the GD θ. MULTI-OUTPUT: the effective samples are the
    #    M = N·d_out outputs (residual / Jacobian-row dimension), so cifar10 (d=10) / sorting-GPT (d=seqlen) work too. ──
    M16 = Nfull * outD                                                   # effective samples = N·d_out
    if (P.get("s24", 0) and start == 0 and _TL.loss.name == "mse"
            and M16 <= 256 and M16 * p <= 300_000_000 and Nfull <= 64):  # any arch; gate on M=N·d (jac_cols = M backward) + M·p memory
        th16 = th.double(); X16 = Xpool.double() if Xpool.is_floating_point() else Xpool   # start==0 only (no §16 re-stream on resume)
        Y16 = Ypool.double() if Ypool.is_floating_point() else Ypool
        Xt16 = Yt16 = None
        if dataset == "chebyshev":                                       # §16 Panel 5 held-out test set (grid midpoints)
            Xt16, Yt16 = _sec16_chebyshev_testset(Nfull, P.get("degree", 3), torch.float64)
        else:                                                            # real held-out images (cifar/mnist) or iid-Gaussian (synthetic/const); None for sorting
            Xt16, Yt16 = _sec16_holdout(dataset, Nfull, P, inD, outD)
        for rec16 in _sec16_driver(th16, X16, Y16, lr, P.get("s24warm", 5), P.get("s24iter", 250),
                                   n, min(p, max(4 * P.get("s24k", 32) + 32, 2 * n + 16)), P.get("seed", 0), 1e-12,
                                   P.get("s24grid", 0.1), P.get("s24ares", 0.01),
                                   Xt16, Yt16, P.get("s24base", 0), P.get("s24k", 32)):
            if mytok != RUN_TOKEN.get(_devkey, mytok):
                return
            yield {"type": "g16", **rec16}

    # ── §17: PER-SAMPLE function-Hessian variant of §16 (each sample uses the top/bottom eigvec of its OWN Q_k=∇²f_k).
    #    Per-sample Lanczos every iteration ⇒ O(M²) jac evals (HEAVY at the ceiling), gated to M≤256 like §16 — the
    #    A100 (80GB) target. The M·p memory ceiling + N≤64 keep it bounded. ──
    if (P.get("s25", 0) and start == 0 and _TL.loss.name == "mse"
            and M16 <= 256 and M16 * p <= 300_000_000 and Nfull <= 64):
        th17 = th.double(); X17 = Xpool.double() if Xpool.is_floating_point() else Xpool
        Y17 = Ypool.double() if Ypool.is_floating_point() else Ypool
        Xt17 = Yt17 = None
        if dataset == "chebyshev":
            Xt17, Yt17 = _sec16_chebyshev_testset(Nfull, P.get("degree", 3), torch.float64)
        else:
            Xt17, Yt17 = _sec16_holdout(dataset, Nfull, P, inD, outD)
        for rec17 in _sec17_driver(th17, X17, Y17, lr, P.get("s24warm", 5), P.get("s24iter", 250),
                                   n, min(p, max(4 * P.get("s24k", 32) + 32, 2 * n + 16)), P.get("seed", 0), 1e-12,
                                   P.get("s24grid", 0.1), P.get("s24ares", 0.01),
                                   Xt17, Yt17, P.get("s25base", 0), P.get("s24k", 32)):
            if mytok != RUN_TOKEN.get(_devkey, mytok):
                return
            yield {"type": "g17", **rec17}

    for t in range(steps + 1):
        if mytok != RUN_TOKEN.get(_devkey, mytok):
            return            # a newer /run started → drop this stale stream so it stops using the GPU
        if not full_batch:    # fresh minibatch for this step (GD + diagnostics); §4d groups undefined
            import numpy as _np
            sel = torch.as_tensor(
                _np.random.RandomState((P["seed"] * 1000003 + t) & 0x7FFFFFFF).choice(Nfull, size=bs, replace=False),
                dtype=torch.long, device=_dev())
            X, Y = Xpool[sel], Ypool[sel]
            pos_rows = neg_rows = torch.empty(0, dtype=torch.long, device=_dev())
        if t == start:
            t0 = time.time()  # reset rate clock once streaming begins (exclude fast-forward)
        if t >= start and t % ee == 0:
            J, out = gradF(th, X)
            loss = float(_TL.loss.value(out, Y, N))
            if not math.isfinite(loss) or loss > 1e10:
                yield {"type": "error",
                       "error": f"diverged at step {t} (loss={loss:.2e}). Lower lr or init_scale."}
                return
            # CE, and the OWT language model under ANY loss, have no scalar per-output residual y−f
            # (Y are token ids and out is (N,T,V)); the §1 residual trace / §4d sign-groups are empty.
            r = None if (ce or dataset == "owt") else (Y - out).reshape(-1)
            cS = _TL.loss.resid_cotangent(out, Y, N)

            needVecs = s4 or s6 or s11    # §4d also projects onto the H eigvecs (match the browser)
            needVals = s1 or s2 or s3 or needVecs
            feTop, feBot, feTV, feBV = [], [], None, None
            if needVecs:
                feTV, feBV, feTop, feBot = lanczos_extreme(
                    lambda v: hvpF(th, X, v), p, nSub, mF, 0xA53F9)
            elif needVals:
                feTop, feBot = lanczos_extreme_vals(
                    lambda v: hvpF(th, X, v), p, n, mV, 0xA53F9)

            hlt = hlb = None
            if s1 or s2 or s3:
                hlt, hlb = lanczos_extreme_vals(
                    lambda v: hvpL(th, X, Y, v), p, n, mV, 0x5EED1)
            sharp = hlt[0] if hlt else None

            gnt = gnb = srt = srb = None
            if (s2 or s3) and gs:
                gnt, gnb = lanczos_extreme_vals(
                    lambda v: hvpG(th, X, v), p, n, mV, 0x6A17C)
                srt, srb = lanczos_extreme_vals(
                    lambda v: hvpS(th, X, v, cS), p, n, mV, 0x7B23D)

            # sign-track H eigvecs once (shared by §4 and §4d)
            if needVecs and feTV is not None:
                for i in range(nSub):
                    feTV[i] = sign_to(feTV[i], prevTop[i]); prevTop[i] = feTV[i]
                    feBV[i] = sign_to(feBV[i], prevBot[i]); prevBot[i] = feBV[i]

            # §4 J onto top-/bottom-n eigvecs of H (raw + Frobenius-normalized)
            jt = jb = jtN = jbN = None
            if s4 and feTV is not None:
                jn = max(float(J.norm()), 1e-30)
                jt = [float(J @ feTV[i]) for i in range(n)]
                jb = [float(J @ feBV[i]) for i in range(n)]
                jtN = [x / jn for x in jt]
                jbN = [x / jn for x in jb]

            # §6 eigenspace rotation
            paPos = paNeg = None
            dimPos = dimNeg = None
            if s6 and feTV is not None:
                posS = pos_subspace(feTop, feTV, efrac)
                negS = neg_subspace(feBot, feBV, efrac)
                paPos = principal_angles(prevPos, posS) if prevPos is not None else None
                paNeg = principal_angles(prevNeg, negS) if prevNeg is not None else None
                prevPos, prevNeg = posS, negS
                dimPos, dimNeg = len(posS), len(negS)

            # ---- multi-sample sections: shared Jacobian columns Jc (M, p), residual rr (M,) ----
            Jc = rr = None
            if ((multi_ok or s12single) and (s7 or s8 or s9 or s10 or s11 or s12 or s13 or s15 or s16 or s17 or s18 or s19 or s20 or s21 or s22)) or (s23 and N <= grid3dcap):   # §15 also runs for a single sample
                Jc, out_flat = jac_cols(th, X)
                rr = (-N * _TL.loss.resid_cotangent(out, Y, N)).reshape(-1)   # generic residual: Y−f (MSE), onehot−softmax (CE)

            # §7 NTK + function-Hessian tensor SVD
            ntkR = ntkH = fhEvT = fhEvB = None
            ntkGs = ntkGsA = ntkGl = ntkGlA = None      # §7a Δλ·Δr products (sync + lagged) and running avgs
            jhe1 = jhe2 = jhe1b = jhe2b = jh2e1 = jh2e2 = jh2e1b = jh2e2b = None
            if (s7 or s13) and multi_ok:
                Kv, Vk = sym_eig_desc(Jc @ Jc.t())                 # NTK eigen
                for k in range(n7):
                    Vk[:, k] = sign_to(Vk[:, k], prevNtk[k]); prevNtk[k] = Vk[:, k]
                GH = torch.zeros(M, M, dtype=DTYPE, device=_dev())  # Hessian Frobenius Gram (Hutchinson)
                for pr in range(nProbe):
                    Hz = jac_hvp(th, X, _randn_vec(p, (0xC0FFEE + pr * 40503) & 0xFFFFFFFF))
                    GH += Hz @ Hz.t()
                GH /= nProbe
                Hv, Vh = sym_eig_desc(GH)                          # right singular vecs of FH tensor
                for k in range(n7):
                    Vh[:, k] = sign_to(Vh[:, k], prevVH[k]); prevVH[k] = Vh[:, k]
                # §7a NTK alignment (always-on panel): residual→NTK eigvec; NTK eigvec→FH right-sing vec
                ntkR = [float(rr @ Vk[:, k]) for k in range(n7)]
                ntkH = [float(Vk[:, 0] @ Vh[:, k]) for k in range(n7)]
                # §7a products: Δsharp(t)·Δ⟨r,vₖ⟩ — synchronous and 1-step lagged — each with a running time-avg.
                # Δx(t)=x(t)-x(t-1).  gₖ=Δλ(t)·Δrₖ(t);  g'ₖ=Δλ(t)·Δrₖ(t-1) (residual change offset back one step).
                dSh = (sharp - prevSharp7a) if (prevSharp7a is not None and sharp is not None) else None
                if dSh is not None and prevNtkR7a is not None:
                    cntGs7a += 1; ntkGs = []; ntkGsA = []
                    for k in range(n7):
                        g = dSh * (ntkR[k] - prevNtkR7a[k]); sumGsync[k] += g
                        ntkGs.append(g); ntkGsA.append(sumGsync[k] / cntGs7a)
                if dSh is not None and prevPrevNtkR7a is not None:
                    cntGl7a += 1; ntkGl = []; ntkGlA = []
                    for k in range(n7):
                        g = dSh * (prevNtkR7a[k] - prevPrevNtkR7a[k]); sumGlag[k] += g
                        ntkGl.append(g); ntkGlA.append(sumGlag[k] / cntGl7a)
                prevPrevNtkR7a = prevNtkR7a; prevNtkR7a = list(ntkR); prevSharp7a = sharp
                if s7 and eigTick % heavyevery == 0:              # heavy §7 FH-eigenvector projections (throttled)
                    UJ = []                                        # left singular vecs of J (param space)
                    for k in range(n7):
                        u = (Jc.t() @ Vk[:, k]) / math.sqrt(max(float(Kv[k]), 1e-30))
                        u = sign_to(u, prevUJ[k]); prevUJ[k] = u; UJ.append(u)
                    nM = min(2, max(1, p // 2))
                    fe1t, fe1b, fe1tv, fe1bv = lanczos_extreme(
                        lambda v: hvpS(th, X, v, Vh[:, 0].reshape(N, outD)), p, nM, mV, 0x9E37)
                    gh1 = Vh[:, 1] if Vh.shape[1] > 1 else Vh[:, 0]
                    fe2t, fe2b, fe2tv, fe2bv = lanczos_extreme(
                        lambda v: hvpS(th, X, v, gh1.reshape(N, outD)), p, nM, mV, 0x7C19)
                    def _l(a, i):
                        return a[i] if i < len(a) else a[0]
                    m1 = [fe1t[0], _l(fe1t, 1), fe1b[0], _l(fe1b, 1)]
                    m2 = [fe2t[0], _l(fe2t, 1), fe2b[0], _l(fe2b, 1)]
                    for j in range(4):
                        m1[j] = sign_to(m1[j], prevM1[j]); prevM1[j] = m1[j]
                        m2[j] = sign_to(m2[j], prevM2[j]); prevM2[j] = m2[j]
                    fhEvT = [fe1tv[0], _l(fe1tv, 1), fe2tv[0], _l(fe2tv, 1)]
                    fhEvB = [fe1bv[0], _l(fe1bv, 1), fe2bv[0], _l(fe2bv, 1)]
                    jhe1 = [float(UJ[k] @ m1[0]) for k in range(n7)]
                    jhe2 = [float(UJ[k] @ m1[1]) for k in range(n7)]
                    jhe1b = [float(UJ[k] @ m1[2]) for k in range(n7)]
                    jhe2b = [float(UJ[k] @ m1[3]) for k in range(n7)]
                    jh2e1 = [float(UJ[k] @ m2[0]) for k in range(n7)]
                    jh2e2 = [float(UJ[k] @ m2[1]) for k in range(n7)]
                    jh2e1b = [float(UJ[k] @ m2[2]) for k in range(n7)]
                    jh2e2b = [float(UJ[k] @ m2[3]) for k in range(n7)]

            # §8 vec(J) onto right singular vecs of the p×(p·dₙ) FH reshape (heaviest section → throttled)
            g2J = g2Jn = None
            if s8 and multi_ok and eigTick % heavyevery == 0:
                jfn = max(float(Jc.norm()), 1e-30)                 # ‖J‖_F
                wv = torch.zeros(p, dtype=DTYPE, device=_dev())
                for a in range(M):
                    wv += (grad_out(th + EPS * Jc[a], X, a)
                           - grad_out(th - EPS * Jc[a], X, a)) / (2 * EPS)
                gt, _, gtv, _ = lanczos_extreme(lambda v: hvpG2(th, X, v), p, n8, mV, 0xB2D4)
                g2J = []
                for k in range(n8):
                    uk = sign_to(gt[k], prevG2[k]); prevG2[k] = uk
                    g2J.append(float(wv @ uk) / math.sqrt(max(gtv[k], 1e-30)))
                g2Jn = [x / jfn for x in g2J]

            # ---- §4b/§4c/§4d base: NTK top eigvec u₁ (deterministic sign), J·r ----
            u1s = Jrs = bEk_vecs = bEk_vals = None
            Jrns = Rn = 1.0
            if multi_ok and (s9 or s10 or s11 or s12 or s15 or s16 or s17 or s18):
                Kb, Vb = sym_eig_desc(Jc @ Jc.t())
                for k in range(min(n, Vb.shape[1])):   # NTK is M×M → only M eigvecs exist (guard M < neig, e.g. tiny §11 runs)
                    Vb[:, k] = pin_sign(Vb[:, k])
                bEk_vecs = Vb; bEk_vals = Kb
                u1s = Vb[:, 0]
                Rn = max(float(rr.norm()), 1e-30)
                Jrs = Jc.t() @ rr
                Jrns = max(float(Jrs.norm()), 1e-30)

            # Q[u₁] eigvecs (shared by §4b and §4d), sign-tracked once
            qeT = qeB = None
            if multi_ok and (s9 or s11):
                qeT, qeB, _, _ = lanczos_extreme(
                    lambda v: hvpS(th, X, v, u1s.reshape(N, outD)), p, n, mV, 0xC0DE1)
                for k in range(n):
                    qeT[k] = sign_to(qeT[k], prevQ9t[k]); prevQ9t[k] = qeT[k]
                    qeB[k] = sign_to(qeB[k], prevQ9b[k]); prevQ9b[k] = qeB[k]

            # §4b J·r onto eigvecs of Q[u₁]; alignment with GN top eigvec g₁=J u₁/‖J u₁‖
            q9t = q9b = q9tN = q9bN = q9tR = q9bR = q9gt = q9gb = None
            if s9 and multi_ok:
                g1 = Jc.t() @ u1s
                g1 = g1 / max(float(g1.norm()), 1e-30)
                q9t = [float(Jrs @ qeT[k]) for k in range(n)]
                q9b = [float(Jrs @ qeB[k]) for k in range(n)]
                q9tN = [x / Jrns for x in q9t]; q9bN = [x / Jrns for x in q9b]
                q9tR = [x / Rn for x in q9t]; q9bR = [x / Rn for x in q9b]
                q9gt = [float(g1 @ qeT[k]) for k in range(n)]
                q9gb = [float(g1 @ qeB[k]) for k in range(n)]

            # §4c Q[u₁]·(J·r) onto GN eigvecs g_k = J v_k/‖·‖
            q10 = q10N = None
            if s10 and multi_ok:
                QJr = hvpS(th, X, Jrs, u1s.reshape(N, outD))
                qn = max(float(QJr.norm()), 1e-30)
                q10 = []
                for k in range(min(n, bEk_vecs.shape[1])):    # ≤ M GN eigvecs exist (clamp so N·d_out<neig doesn't index OOB)
                    g = Jc.t() @ bEk_vecs[:, k]
                    g = g / max(float(g.norm()), 1e-30)
                    q10.append(float(QJr @ g))
                q10N = [x / qn for x in q10]

            # §4d per-group J / J·r onto the shared H and Q[u₁] eigvecs
            d4Ht = d4Hb = d4Qt = d4Qb = None
            if s11 and multi_ok and feTV is not None and qeT is not None:
                def _grp(rows, weighted):
                    if rows.numel() == 0:
                        return torch.zeros(p, dtype=DTYPE, device=_dev())
                    sub = Jc[rows]
                    if weighted:
                        sub = rr[rows].unsqueeze(1) * sub
                    return sub.sum(0)
                Jp, Jn_ = _grp(pos_rows, False), _grp(neg_rows, False)
                Jrp, Jrn = _grp(pos_rows, True), _grp(neg_rows, True)
                d4Ht = [float(Jp @ feTV[k]) for k in range(n)] + [float(Jn_ @ feTV[k]) for k in range(n)]
                d4Hb = [float(Jp @ feBV[k]) for k in range(n)] + [float(Jn_ @ feBV[k]) for k in range(n)]
                d4Qt = [float(Jrp @ qeT[k]) for k in range(n)] + [float(Jrn @ qeT[k]) for k in range(n)]
                d4Qb = [float(Jrp @ qeB[k]) for k in range(n)] + [float(Jrn @ qeB[k]) for k in range(n)]

            # §9 theory vs empirical σ₁ over frozen-Q windows (η/N effective step). thP=predicted, thA=actual.
            # thPpsd = thP plus the dropped 2nd-order PSD term Σ‖ΔJᵀu₁‖² (§9b panels).
            # thAH = the FULL loss-Hessian sharpness λmax(∇²L)=λmax(G+S) — §9c compares the predictions
            # against it instead of the Gauss-Newton edge thA (=λmax(G)); the gap is the residual term S.
            thP = thA = thPpsd = None
            thP_d = thPpsd_d = None       # §9d / §9d-c: predictions with the quad-self-computed residual
            thAH = None
            if s14 or s16 or s17:         # §9c/§9d-c/§10 actual: full loss-Hessian sharpness λmax(∇²L) (reuse §1's if present)
                thAH = sharp if sharp is not None else float(lanczos_extreme_vals(
                    lambda v: hvpL(th, X, Y, v), p, 1, mV, 0x5EED1)[0][0])
            if s12 or s14 or s15 or s16:
                etaN = lr / max(N, 1); reps_ = max(1, ee)
                if thT0 < 0 or (t - thT0) >= qapprox:               # window start: freeze θ₀, J₀, FH; reset accumulators
                    thT0 = t; thTh0 = th.clone(); thAcc1 = 0.0; thAcc2 = 0.0; thProd3 = 1.0; thProd4 = 1.0; thProd5 = 1.0
                    thAccPSD = 0.0; thAccPSD1 = 0.0
                    if multi_ok:
                        thJ, _ = jac_cols(th, X)                  # predicted Jacobian J₀ (M, p)
                        thFroz = fh_frozen(th, X, nProbe, min(n, M), 2, mV, M)
                        thBase = float(_safe_eigvalsh(thJ @ thJ.t())[-1])
                        if s15 or s16:    # §9d: freeze f₀,J₀; reset quad-GD displacement & parallel accumulators
                            thJ0 = thJ.clone(); thF0 = (Y.reshape(-1) - rr).clone() if rr is not None else None
                            thDth = torch.zeros(p, dtype=DTYPE, device=_dev()); thJ_d = thJ.clone()
                            thProd3_d = 1.0; thProd4_d = 1.0; thProd5_d = 1.0; thAcc2_d = 0.0; thAccPSD_d = 0.0
                    elif not multi:
                        Jc0, o0 = gradF(th, X); thJp = Jc0.clone(); thBase = float(Jc0 @ Jc0)
                        if s15 or s16:    # §9d single-sample: freeze f₀,J₀; reset displacement & accumulators
                            thJp0 = Jc0.clone(); thF0s = float(o0.reshape(-1)[0])
                            thDth_s = torch.zeros(p, dtype=DTYPE, device=_dev()); thJp_d = Jc0.clone()
                            thAcc1_d = 0.0; thAccPSD1_d = 0.0
                thP = [None]*5; thA = [None]*5      # display cols 1-5: Eq-13, Eq-21, Eq-22, Eq-23, Eq-29 (col-4=thProd5=Eq-23, col-5=thProd4=Eq-29)
                thPpsd = [None]*5   # §9b: prediction + accumulated 2nd-order PSD term ‖ΔJᵀu₁‖²
                if multi_ok and thJ is not None and thFroz is not None and bEk_vals is not None:
                    Jt = thJ; F = thFroz
                    Kw, Vw = sym_eig_desc(Jt @ Jt.t())                  # propagated NTK eigen
                    NV0 = min(n, M)
                    for k in range(NV0):
                        Vw[:, k] = pin_sign(Vw[:, k])
                    sig1 = max(float(Kw[0]), 1e-30); sgT = math.sqrt(sig1); u1 = Vw[:, 0]
                    v1 = Jt.t() @ u1; v1 = v1 / max(float(v1.norm()), 1e-30)
                    sigAct = float(bEk_vals[0]); thA[1] = thA[2] = thA[3] = thA[4] = sigAct
                    cap = 10*max(sigAct, 1e-30); clmp = lambda x: min(max(x, 0.0), cap)
                    def gnv(k):
                        g = Jt.t() @ Vw[:, k]; return g / max(float(g.norm()), 1e-30)
                    def fhBil(vk):
                        s = 0.0
                        for i in range(len(F["gamma"])):
                            yu = float(F["y"][i] @ u1)
                            sj = sum(F["tau"][i][j]*float(v1 @ F["z"][i][j])*float(vk @ F["z"][i][j]) for j in range(len(F["z"][i])))
                            s += F["gamma"][i]*sj*yu
                        return s
                    thP[1] = clmp(thBase + thAcc2)                      # col-2 (Eq-21): additive first-order Δσ (flips with residual sign)
                    thP[2] = clmp(thBase*thProd3); thP[3] = clmp(thBase*thProd5); thP[4] = clmp(thBase*thProd4)   # col-4=Eq-23, col-5=Eq-29
                    # §9b: same predictions + the dropped 2nd-order PSD term Σ‖ΔJᵀu₁‖² (always ≥0 — a sharpening floor)
                    thPpsd[1] = clmp(thBase + thAcc2 + thAccPSD)
                    thPpsd[2] = clmp(thBase*thProd3 + thAccPSD); thPpsd[3] = clmp(thBase*thProd5 + thAccPSD)
                    thPpsd[4] = clmp(thBase*thProd4 + thAccPSD)
                    qv1 = hvpS(thTh0, X, v1, u1.reshape(N, outD))      # Q[u₁]v₁ = (Σ_a u₁_a ∇²f_a)·v₁ via the weighted HVP (2 grad
                    #   evals, not the full M×p jac_hvp) ⇒ the EXACT bilinear v₁ᵀQ[u₁]v_k = qv1·v_k (used by Eq-22 & Eq-23)
                    pproj = float(rr @ u1)                             # col-3 (Eq-22): p_t = r·u₁ (residual onto NTK mode u₁ = v₁-coeff of Jᵀr)
                    thProd3 *= (1 + 2 * etaN * pproj * float(qv1 @ v1)) ** reps_   # σ_{t+1}=σ_t[1+2η (r·u₁) v₁ᵀQ[u₁]v₁]  (exact bilinear)
                    S4 = 0.0; S5 = 0.0; NV = min(max(tset, 1), NV0)    # both sum Σ_k (√σ_k/√σ₁)(r·u_k)·v₁ᵀQ[u₁]v_k over the top-|T| modes
                    for vk in range(NV):
                        sgv = math.sqrt(max(float(Kw[vk]), 1e-30)); rho = float(rr @ Vw[:, vk]); gk = gnv(vk)
                        S4 += (sgv/sgT) * fhBil(gk) * rho              # col-5 (Eq-29): bilinear via the γτz eigendecomposition of Q (approx)
                        S5 += (sgv/sgT) * float(qv1 @ gk) * rho        # col-4 (Eq-23): bilinear computed directly (exact — no Q decomposition)
                    thProd4 *= (1 + 2 * etaN * S4) ** reps_            # col-5 (Eq-29)
                    thProd5 *= (1 + 2 * etaN * S5) ** reps_            # col-4 (Eq-23): differs from Eq-29 only by the Q-eigendecomposition approx
                    for _ in range(reps_):                             # advance Eq-15: J += (η/N) Q[Jᵀr], Q frozen at θ₀
                        QW = jac_hvp(thTh0, X, thJ.t() @ rr)      # {∇²f_a·(Jᵀr)} at θ₀
                        qu = (u1.unsqueeze(1) * QW).sum(0)              # Q[u₁](Jᵀr) = Σ_a u₁_a QW[a]
                        thAcc2 += 2 * etaN * sgT * float(v1 @ qu)       # col-2 Δσ (first-order, flips with residual sign)
                        thAccPSD += (etaN ** 2) * float(qu @ qu)        # §9b: 2nd-order PSD term ‖ΔJᵀu₁‖² = (η/N)²‖qu‖²
                        thJ = thJ + etaN * QW
                elif not multi and thJp is not None:
                    # col-1 (Eq-13): additive first-order Δσ = 2η r (JᵀQ J) on propagated J — flips with residual sign
                    Jcur, ocur = gradF(th, X); sigAct = float(Jcur @ Jcur); thA[0] = sigAct
                    cap0 = 10*max(sigAct, 1e-30)
                    thP[0] = min(max(thBase + thAcc1, 0.0), cap0); rsc = float((Y.reshape(-1) - ocur)[0])
                    thPpsd[0] = min(max(thBase + thAcc1 + thAccPSD1, 0.0), cap0)   # §9b: + Σ‖ΔJ‖² (≥0)
                    for _ in range(reps_):
                        QJ = hvpF(thTh0, X, thJp)
                        thAcc1 += 2 * lr * rsc * float(thJp @ QJ)
                        thAccPSD1 += ((lr * rsc) ** 2) * float(QJ @ QJ)   # §9b: ‖ΔJ‖² = (η r)²‖QJ‖²
                        thJp = thJp + lr * rsc * QJ

                # ---- §9d: SAME predictions, residual SELF-COMPUTED by the frozen quadratic model (closed loop) ----
                # inside the window r is the quad-GD residual r_q = Y − f_quad(θ₀+Δθ), not the live residual.
                thP_d = [None]*5; thPpsd_d = [None]*5
                if multi_ok and thJ_d is not None and thFroz is not None and bEk_vals is not None and thF0 is not None:
                    Jd = thJ_d; Fz = thFroz; dth = thDth
                    HzD = jac_hvp(thTh0, X, dth)                          # Q[Δθ] = {∇²f_a(θ₀)·Δθ}, (M,p)
                    r_q = Y.reshape(-1) - (thF0 + thJ0 @ dth + 0.5 * (HzD @ dth))   # quad-model residual
                    Kw, Vw = sym_eig_desc(Jd @ Jd.t())
                    NV0 = min(n, M)
                    for k in range(NV0):
                        Vw[:, k] = pin_sign(Vw[:, k])
                    sgT = math.sqrt(max(float(Kw[0]), 1e-30)); u1 = Vw[:, 0]
                    v1 = Jd.t() @ u1; v1 = v1 / max(float(v1.norm()), 1e-30)
                    cap = 10 * max(float(bEk_vals[0]), 1e-30); clmpd = lambda x: min(max(x, 0.0), cap)
                    def gnv_d(k):
                        g = Jd.t() @ Vw[:, k]; return g / max(float(g.norm()), 1e-30)
                    def fhBil_d(vk):
                        s = 0.0
                        for i in range(len(Fz["gamma"])):
                            yu = float(Fz["y"][i] @ u1)
                            sj = sum(Fz["tau"][i][j]*float(v1 @ Fz["z"][i][j])*float(vk @ Fz["z"][i][j]) for j in range(len(Fz["z"][i])))
                            s += Fz["gamma"][i]*sj*yu
                        return s
                    thP_d[1] = clmpd(thBase + thAcc2_d)
                    thP_d[2] = clmpd(thBase * thProd3_d); thP_d[3] = clmpd(thBase * thProd5_d); thP_d[4] = clmpd(thBase * thProd4_d)
                    thPpsd_d[1] = clmpd(thBase + thAcc2_d + thAccPSD_d)
                    thPpsd_d[2] = clmpd(thBase * thProd3_d + thAccPSD_d); thPpsd_d[3] = clmpd(thBase * thProd5_d + thAccPSD_d)
                    thPpsd_d[4] = clmpd(thBase * thProd4_d + thAccPSD_d)
                    qv1 = hvpS(thTh0, X, v1, u1.reshape(N, outD))
                    pproj = float(r_q @ u1)
                    thProd3_d *= (1 + 2 * etaN * pproj * float(qv1 @ v1)) ** reps_
                    S4 = 0.0; S5 = 0.0; NV = min(max(tset, 1), NV0)
                    for vk in range(NV):
                        sgv = math.sqrt(max(float(Kw[vk]), 1e-30)); rho = float(r_q @ Vw[:, vk]); gk = gnv_d(vk)
                        S4 += (sgv/sgT) * fhBil_d(gk) * rho
                        S5 += (sgv/sgT) * float(qv1 @ gk) * rho
                    thProd4_d *= (1 + 2 * etaN * S4) ** reps_
                    thProd5_d *= (1 + 2 * etaN * S5) ** reps_
                    for _ in range(reps_):                                # advance quad GD (Eq-15 with r_q) + Δθ
                        g = thJ_d.t() @ r_q
                        QW = jac_hvp(thTh0, X, g)
                        qu = (u1.unsqueeze(1) * QW).sum(0)
                        thAcc2_d += 2 * etaN * sgT * float(v1 @ qu)
                        thAccPSD_d += (etaN ** 2) * float(qu @ qu)
                        thJ_d = thJ_d + etaN * QW
                        thDth = thDth + etaN * g
                elif (not multi) and thJp_d is not None:
                    Qd = hvpF(thTh0, X, thDth_s)                          # ∇²f(θ₀)·Δθ
                    r_qs = float(Y.reshape(-1)[0] - (thF0s + float(thJp0 @ thDth_s) + 0.5 * float(thDth_s @ Qd)))
                    cap0 = 10 * max(float(thJp_d @ thJp_d), 1e-30)
                    thP_d[0] = min(max(thBase + thAcc1_d, 0.0), cap0)
                    thPpsd_d[0] = min(max(thBase + thAcc1_d + thAccPSD1_d, 0.0), cap0)
                    for _ in range(reps_):
                        QJ = hvpF(thTh0, X, thJp_d)
                        thAcc1_d += 2 * lr * r_qs * float(thJp_d @ QJ)
                        thAccPSD1_d += ((lr * r_qs) ** 2) * float(QJ @ QJ)
                        thJp_d = thJp_d + lr * r_qs * QJ
                        thDth_s = thDth_s + lr * r_qs * thJp_d

            # ---- §10 CUBIC approximation (Eq-47/51 σ₁ predictions, exact J&Q propagation) ----
            cub = cubic_step(cubCtx, cubSt, th, X, Y, t, J, out, rr, bEk_vals, thAH) if s17 else {}

            # ---- §11 3D grids T1=JᵢᵀQⱼJₖ, T2=uⱼuₖT1, T3=rᵢuⱼuₖT1 (multi-sample, small M, SLQ cadence) ----
            g3d = None
            if (s18 and multi_ok and eigTick % g3dstride == 0 and Jc is not None and u1s is not None
                    and rr is not None and N <= grid3dcap):
                # §11 uses ONE output per sample — the GROUND-TRUTH-LABEL output (argmax of the target). For
                # single-output this is identity (Ms=N); for multi-output (cifar/sorting) it picks f_i[label_i]
                # so (i,j,k) range over the N samples, and the cap applies to N (not N·d_out).
                if outD > 1:
                    lblsel = torch.arange(N, device=_dev()) * outD + Y.reshape(N, outD).argmax(dim=1)
                    Jc11 = Jc[lblsel]; rr11 = rr[lblsel]
                    Kx, Vx = sym_eig_desc(Jc11 @ Jc11.t()); u11 = pin_sign(Vx[:, 0])   # NTK eigvec over the N label-outputs
                else:
                    lblsel = None; Jc11 = Jc; rr11 = rr; u11 = u1s
                Ms = N
                T1 = torch.zeros(Ms, Ms, Ms, dtype=DTYPE, device=_dev())
                Gd = torch.zeros(Ms, Jc11.shape[1], dtype=DTYPE, device=_dev())   # gᵢ = Qᵢ Jᵢ (diagonal row of each HVP)
                for kk in range(Ms):
                    QJk = jac_hvp(th, X, Jc11[kk])               # (M,p): {Q_a · J_sel[k]}_a over ALL model outputs
                    QJr = QJk[lblsel] if lblsel is not None else QJk   # (Ms,p): ground-truth-label Hessian rows
                    T1[:, :, kk] = Jc11 @ QJr.t()                 # [i,j] = Jᵢ·(Qⱼ Jₖ)
                    Gd[kk] = QJr[kk]                              # Qₖ Jₖ  (reused → no extra HVP)
                T2 = T1 * u11.view(1, Ms, 1) * u11.view(1, 1, Ms)
                T3 = T2 * rr11.view(Ms, 1, 1)
                # §11 SQUARE — S[i,j] = Jᵢᵀ Qᵢ Jⱼ · rⱼ = (Qᵢ Jᵢ)·Jⱼ · rⱼ (2D over (i,j); classes i=j / i≠j)
                Smat = (Gd @ Jc11.t()) * rr11.view(1, Ms)        # (Ms,Ms): [i,j] = gᵢ·Jⱼ · rⱼ
                jj_ = torch.arange(Ms, device=_dev())
                cats = torch.where(jj_.view(Ms, 1) == jj_.view(1, Ms), 0, 1).reshape(-1)   # 0:i=j  1:i≠j

                def _evs(S):   # per (i=j / i≠j) class: [mean, std, mean|·|]
                    f = S.reshape(-1)
                    out = []
                    for c in range(2):
                        sel = f[cats == c]
                        out.append([float(sel.mean()), float(sel.std(unbiased=False)), float(sel.abs().mean())]
                                   if int((cats == c).sum()) else [0.0, 0.0, 0.0])
                    return out
                # Per-class evolution stats — mean ± std over the FULL grid (NOT the sparsified subset) for the 5
                # diagonal classes of (i,j,k): 0:i=j=k 1:i=j≠k 2:i≠j=k 3:i=k≠j 4:i≠j≠k. Tiny payload → curves vs step.
                ai = torch.arange(Ms, device=_dev())
                eij = ai.view(Ms, 1, 1) == ai.view(1, Ms, 1)
                ejk = ai.view(1, Ms, 1) == ai.view(1, 1, Ms)
                eik = ai.view(Ms, 1, 1) == ai.view(1, 1, Ms)
                catf = torch.where(eij & ejk, 0, torch.where(eij, 1, torch.where(ejk, 2,
                                   torch.where(eik, 3, 4)))).reshape(-1)

                def _ev(T):   # per class: [mean, std, mean|·|] over the FULL grid
                    f = T.reshape(-1)
                    out = []
                    for c in range(5):
                        sel = f[catf == c]
                        if int((catf == c).sum()):
                            out.append([float(sel.mean()), float(sel.std(unbiased=False)), float(sel.abs().mean())])
                        else:
                            out.append([0.0, 0.0, 0.0])
                    return out

                # Once the cube exceeds the render budget, emit a representative subset that SPANS the value range
                # (top-|value| structure + random bulk) so the colours stay fine-grained. idx = i·M²+j·M+k.
                sparse = (Ms * Ms * Ms) > G3D_MAXPTS

                def _pack(T):
                    return _g3d_pack(T.reshape(-1))

                def _pospct(T):   # 100·(Σ positive)/(Σ positive + |Σ negative|) over the full N³ grid
                    f = T.reshape(-1)
                    ps = float(f[f > 0].sum()); ns = float(f[f < 0].sum())   # ns ≤ 0 → |neg sum| = -ns
                    den = ps - ns
                    return ps / den * 100.0 if den > 1e-30 else 0.0

                v1, ix1 = _pack(T1); v2, ix2 = _pack(T2); v3, ix3 = _pack(T3)
                dd = torch.arange(Ms, device=_dev())   # i=j=k diagonal values (always sent so the highlighted
                d1 = T1[dd, dd, dd]; d2 = T2[dd, dd, dd]; d3 = T3[dd, dd, dd]   # diagonal is coloured even when sparse
                sqv, sqi = _g3d_pack(Smat.reshape(-1))   # the square is Ms² — same span-preserving subset
                sqsparse = sqi is not None
                g3d = {"M": Ms, "sparse": sparse, "t1": v1, "t2": v2, "t3": v3,
                       "d1": d1.cpu().tolist(), "d2": d2.cpu().tolist(), "d3": d3.cpu().tolist(),
                       "pp": [_pospct(T1), _pospct(T2), _pospct(T3)],
                       "ev": {"t1": _ev(T1), "t2": _ev(T2), "t3": _ev(T3)},
                       "sq": sqv, "sqsparse": sqsparse,                        # 4th column: S[i,j], idx=i·M+j
                       "sqd": torch.diagonal(Smat).cpu().tolist(),            # i=j diagonal (highlight; full M)
                       "sqpp": _pospct(Smat), "sqev": _evs(Smat)}             # stats over the FULL square
                if sqsparse:
                    g3d["sqi"] = sqi
                if sparse:
                    g3d.update({"i1": ix1, "i2": ix2, "i3": ix3})

            # §12 — per-sample Hessian Q_i = ∇²f_i (ground-truth-label output); eigenvector cross-similarity
            # grids/cuboids + principal angles over sample pairs (i,j). MATRIX-FREE: extract the top-K12 &
            # bottom-K12 eigenpairs of each Q_i by Lanczos on v↦Q_i·v — no p×p Hessian is formed. For the MLP
            # the N samples' Lanczos run IN ONE BATCH via a vmap'd HVP (a single vectorised forward/backward per
            # step instead of N) — ~4–8× faster on GPU. Other archs fall back to the per-sample loop. Gated on N.
            sec12 = None; sec14 = None; sec13 = None; sec15 = None; sec15n = None; sec15p34 = None; sec15b = None
            if ((s19 or s20 or s21 or s22) and eigTick % g3dstride == 0   # §12/§13/§14 share §11's cube cadence (the N³ cubes dominate file size)
                    and (multi_ok or s12single) and Jc is not None and rr is not None and N <= sec12ncap):
                lblsel12 = (torch.arange(N, device=_dev()) * outD + Y.reshape(N, outD).argmax(dim=1)
                            if outD > 1 else torch.arange(N, device=_dev()))
                K12 = max(1, min(SEC12_KFULL, p // 2))          # top-/bottom-K12 eigenpairs (covers k=1,2,5,10,kfull)
                m12 = min(p, max(6 * K12, 64))                  # Lanczos steps — enough to converge the K12-th eigenpair
                seeds12 = [(0x5EC12 + i * 7919) & 0x7FFFFFFF for i in range(N)]
                if arch == "mlp":
                    import torch.func as _tf
                    lbl12 = (Y.reshape(N, outD).argmax(dim=1) if outD > 1
                             else torch.zeros(N, dtype=torch.long, device=_dev()))
                    OH = torch.zeros(N, outD, dtype=DTYPE, device=_dev())
                    OH[torch.arange(N, device=_dev()), lbl12] = 1.0          # GT-label selector per sample
                    spec12 = _TL.model.spec
                    def _f12(t, x, oh):                                      # scalar = GT-label output of one sample
                        return (fwd(t, spec12, x.unsqueeze(0))[0].reshape(-1) * oh).sum()
                    def _hv12(t, x, oh, v):                                  # ∇²f_i·v via jvp-of-grad (exact)
                        return _tf.jvp(lambda tt: _tf.grad(_f12)(tt, x, oh), (t,), (v,))[1]
                    hvp_batch = lambda V: _tf.vmap(_hv12, in_dims=(None, 0, 0, 0))(th, X, OH, V)
                    TV, TW, BV, BW = batched_lanczos_extreme(hvp_batch, p, N, K12, m12, seeds12)
                else:
                    TVl = []; TWl = []; BVl = []; BWl = []
                    for i in range(N):
                        ci = torch.zeros(N, outD, dtype=DTYPE, device=_dev())
                        ci.view(-1)[int(lblsel12[i])] = 1.0
                        tv, bv, tval, bval = lanczos_extreme(
                            lambda v, ci=ci: hvpS(th, X, v, ci), p, K12, m12, seeds12[i])
                        TVl.append(torch.stack(tv)); BVl.append(torch.stack(bv))
                        TWl.append(torch.tensor(tval, dtype=DTYPE, device=_dev()))
                        BWl.append(torch.tensor(bval, dtype=DTYPE, device=_dev()))
                    TV, TW, BV, BW = torch.stack(TVl), torch.stack(TWl), torch.stack(BVl), torch.stack(BWl)
                r12 = rr[lblsel12]; Jg12 = Jc[lblsel12]
                gTV = gTW = gBV = gBW = None; wbases = None
                if s22:                    # §12b panels 3/4: top-2/bottom-2 eigenpairs of the AVERAGED Hessian (1/N)Σ_i Q_i
                    cAvg = torch.zeros(N, outD, dtype=DTYPE, device=_dev())   # = ∇²((1/N)Σ_i f_i^GT) via hvpS with cotangent 1/N
                    cAvg.view(-1)[lblsel12] = 1.0 / N                         #   (eigvecs are scale-free; the 1/N only rescales σ)
                    gtv, gbv, gtw, gbw = lanczos_extreme(lambda v: hvpS(th, X, v, cAvg), p, 2, m12, 0x6105A1)
                    gTV = torch.stack(gtv); gBV = torch.stack(gbv)
                    gTW = torch.tensor(gtw, dtype=DTYPE, device=_dev()); gBW = torch.tensor(gbw, dtype=DTYPE, device=_dev())
                    # panels 5/6/7: top-2/bottom-2 eigenpairs of the WEIGHTED-AVERAGE Hessian (Σ_i w_i Q_i)/(Σ_i w_i),
                    #   w = |top NTK eigvec| (p5), |2nd NTK eigvec| (p6), unit-normalized random U[0,1] (p7). NTK = J Jᵀ
                    #   (N×N per-sample GT-gradient Gram); |·| makes the weights gauge-free. cotangent = w/Σw ⇒ a true average.
                    ntk = Jg12 @ Jg12.t()                                    # (N,N)
                    nevecs = _safe_eigh(ntk).eigenvectors                    # columns ascending by eigenvalue
                    v1 = nevecs[:, -1].abs()                                 # |top NTK eigvec|
                    v2 = (nevecs[:, -2] if N >= 2 else nevecs[:, -1]).abs()  # |2nd NTK eigvec| (falls back to top at N=1)
                    rrng = mulberry32(0x7A9D07)                              # fixed seed ⇒ reproducible + browser-identical
                    rvec = torch.tensor([rrng() for _ in range(N)], dtype=DTYPE, device=_dev())
                    rvec = rvec / rvec.norm().clamp_min(1e-30)               # unit-normalized random U[0,1]
                    wbases = []
                    for wi, wv in enumerate([v1, v2, rvec]):
                        cW = torch.zeros(N, outD, dtype=DTYPE, device=_dev())
                        cW.view(-1)[lblsel12] = wv / wv.sum().clamp_min(1e-30)   # weighted-average cotangent (Σ c_i = 1)
                        wtv, wbv, wtw, wbw = lanczos_extreme(lambda v, c=cW: hvpS(th, X, v, c), p, 2, m12, (0x7B0001 + wi * 0x101))
                        wbases.append({"tv": torch.stack(wtv), "tw": torch.tensor(wtw, dtype=DTYPE, device=_dev()),
                                       "bv": torch.stack(wbv), "bw": torch.tensor(wbw, dtype=DTYPE, device=_dev())})
                if s19 or s22:             # §12a (angles+diag-align) ⊕ §12b (proj) — both in one payload, shown by their toggles
                    sec12 = _sec12_payload(TV, TW, BV, BW, r12, Jg12, grid3dcap, gTV=gTV, gTW=gTW, gBV=gBV, gBW=gBW, wbases=wbases)
                if s21 and multi_ok and N <= grid3dcap:       # §13 (own toggle): exact ref J_iᵀQ_jJ_k·r_k (N HVPs, §11's tool) + G1/G2/G3 — multi-only (N³ cube)
                    T1_13 = torch.zeros(N, N, N, dtype=DTYPE, device=_dev())
                    QJ_list = []
                    for kk in range(N):
                        QJk = jac_hvp(th, X, Jg12[kk])                  # (M,p) = {Q_a·J_k}_a over all outputs
                        QJr = QJk[lblsel12] if outD > 1 else QJk        # (N,p) GT-label rows = {Q_i·J_kk}_i
                        QJ_list.append(QJr)
                        T1_13[:, :, kk] = Jg12 @ QJr.t()                # [i,j] = J_i·(Q_j J_k)
                    ref13 = T1_13 * r12.view(1, 1, N)                   # exact reference · r_k
                    cur13 = {"TV": TV, "TW": TW, "BV": BV, "BW": BW, "r": r12, "Jg": Jg12}
                    if sec13_prev is not None:                 # need t-1 for Q_i/Q_k & J',r → skip the first eig-tick
                        sec13 = _sec13_stats(cur13, sec13_prev, ref13, lr)
                        if sec13_qjprev is not None:           # panel 4: residual-reweighting asymmetry ‖A−B‖/‖A‖·100
                            QJp, rpv = sec13_qjprev["QJ"], sec13_qjprev["r"]   # QJp[j,i]=Q_{i,t}J_{j,t} ; rpv=r_{·,t}, all at PREV tick t
                            A = torch.einsum('jip,i,j->p', QJp, r12, rpv)      # (Σ_i r_{i,t+1}Q_{i,t})(Σ_j J_{j,t} r_{j,t})
                            B = torch.einsum('jip,i,j->p', QJp, rpv, r12)      # (Σ_i r_{i,t}Q_{i,t})(Σ_j J_{j,t} r_{j,t+1})
                            nA = float(A.norm())
                            sec13["ndiff"] = (float((A - B).norm()) / nA * 100.0) if nA > 0 else 0.0
                    sec13_prev = {kk_: vv_.detach() for kk_, vv_ in cur13.items()}
                    sec13_qjprev = {"QJ": torch.stack(QJ_list).detach(), "r": r12.detach()}   # (N_j,N_i,p) Q_iJ_j at this tick → PREV for next
                if s20 and multi_ok:                          # §14 shares §12's per-sample eigenpairs — multi-only (N³ cubes)
                    sec14_rhist.append(r12.detach())
                    if len(sec14_rhist) > 5:
                        sec14_rhist.pop(0)
                    cur14 = {"TV": TV, "TW": TW, "BV": BV, "BW": BW, "r": r12, "Jg": Jg12}
                    if sec14_prev is not None:                 # need t (prev) for u,σ,r_j,J_k; skip the first eig-tick (like §13)
                        sec14 = _sec14_payload(sec14_prev, cur14, lr, grid3dcap, sec14_rhist)
                    sec14_prev = {kk_: vv_.detach() for kk_, vv_ in cur14.items()}

            if s23 and (N * outD) <= grid3dcap and Jc is not None and rr is not None:   # §15: 2nd-difference decomposition (multi OR single sample; effective samples M=N·d_out, A/B are M×M)
                M15 = N * outD                                                # EFFECTIVE samples = N·d_out
                lblsel15 = torch.arange(M15, device=_dev())                   # ALL N·d_out outputs are the effective samples (not just the GT-label)
                # §15 in float64: the FD 2nd-difference D² is precision-critical at tiny ‖J‖² scales (e.g. chebyshev), where
                # float32 cancellation noise dominates the divergence. MLP forward is dtype-pure ⇒ safe to recompute in f64.
                f64_15 = isinstance(_TL.model, MlpModel) and DTYPE != torch.float64
                X15 = X.double() if f64_15 else X
                if f64_15:
                    if sec15_th64 is None:
                        sec15_th64 = th.double()          # seed the float64 shadow at the current θ (θ₀ on the first §15 tick)
                    th15 = sec15_th64                     # evaluate §15 on the float64 shadow trajectory (clean D²)
                else:
                    th15 = th                             # DTYPE already float64 (CPU server) ⇒ displayed trajectory is exact
                Jg15 = (_TL.model.jac_cols(th15, X15)[0][lblsel15] if f64_15 else Jc[lblsel15])
                r15 = (rr[lblsel15].double() if f64_15 else rr[lblsel15])
                # ---- panel 5: per-sample NORM evolution (mean ± std across the N samples), at the current step t ----
                jn15 = Jg15.norm(dim=1); rn15 = r15.abs(); gn15 = rn15 * jn15   # per-effective-sample (M) norms
                acc15 = torch.zeros(M15, dtype=DTYPE, device=_dev())   # ‖Q_{t,k}‖_F via Hutchinson: E_v‖Q_k v‖²=tr(Q_k²)
                for pr in range(4):
                    Qv15 = jac_hvp(th15, X15, _randn_vec(p, (0x5EC15 + pr * 0x9E3779B1) & 0xFFFFFFFF))[lblsel15]
                    acc15 = acc15 + (Qv15 * Qv15).sum(dim=1)
                hn15 = torch.sqrt(acc15 / 4.0)
                _ms15 = lambda x: [float(x.mean()), float(x.std(unbiased=False))]
                sec15n = {"g": _ms15(gn15), "j": _ms15(jn15), "h": _ms15(hn15), "r": _ms15(rn15)}
                # ---- new Panel B: per-EFFECTIVE-sample top/bottom-eigvec projection ratio of ∇f_k on its own Q_k (k = one of M=N·d outputs) ----
                mB15 = min(p, 48); seedsB15 = [(0x5EC15B + k * 7919) & 0x7FFFFFFF for k in range(M15)]
                if arch == "mlp":
                    import torch.func as _tf
                    XrepB = X15.repeat_interleave(outD, dim=0)                 # (M,indim): input for effective sample k = X[k//d_out]
                    OHB15 = torch.eye(outD, dtype=X15.dtype, device=_dev()).repeat(N, 1)   # (M,d_out): row k one-hot at output c=k%d_out
                    specB15 = _TL.model.spec
                    def _fB15(tt, x, oh): return (fwd(tt, specB15, x.unsqueeze(0))[0].reshape(-1) * oh).sum()
                    def _hvB15(tt, x, oh, v): return _tf.jvp(lambda q: _tf.grad(_fB15)(q, x, oh), (tt,), (v,))[1]
                    hvpB15 = lambda V: _tf.vmap(_hvB15, in_dims=(None, 0, 0, 0))(th15, XrepB, OHB15, V)
                    TVb15, TWb15, BVb15, BWb15 = batched_lanczos_extreme(hvpB15, p, M15, 1, mB15, seedsB15, dt=X15.dtype)
                else:                                                         # non-MLP (cifar conv / sorting GPT): per-effective-sample Lanczos with a single-output cotangent
                    TVl = []; TWl = []; BVl = []; BWl = []
                    for k in range(M15):
                        ck = torch.zeros(N * outD, dtype=X15.dtype, device=_dev()); ck[k] = 1.0; ck = ck.reshape(N, outD)
                        tv, bv, tw, bw = lanczos_extreme(lambda v, ck=ck: hvpS(th15, X15, v, ck), p, 1, mB15, seedsB15[k], dt=X15.dtype)
                        TVl.append(torch.stack(tv)); BVl.append(torch.stack(bv))
                        TWl.append(torch.tensor(tw, dtype=X15.dtype, device=_dev())); BWl.append(torch.tensor(bw, dtype=X15.dtype, device=_dev()))
                    TVb15, TWb15, BVb15, BWb15 = torch.stack(TVl), torch.stack(TWl), torch.stack(BVl), torch.stack(BWl)
                sec15b = _sec15_panelB(TVb15[:, 0], BVb15[:, 0], TWb15[:, 0], BWb15[:, 0], Jg15)
                if (len(sec15_hist) >= 2 and sec15_hist[-1]["t"] == t - 1 and sec15_hist[-2]["t"] == t - 2):   # 3 CONSECUTIVE GD steps (eigevery=1): the per-step 2nd-difference theory requires it
                    tm1, tm2 = sec15_hist[-1], sec15_hist[-2]
                    hvp1 = lambda v: jac_hvp(tm1["th"], X15, v)[lblsel15]
                    hvp2 = lambda v: jac_hvp(tm2["th"], X15, v)[lblsel15]
                    sec15, Amat15, Bmat15, u1_15 = _sec15_stats(hvp1, hvp2, Jg15, tm1["J"], tm2["J"], tm1["r"], tm2["r"], lr, N, diveps=P.get("divreg", 0.3))
                    hvp_at15 = lambda thx, v: jac_hvp(thx, X15, v)[lblsel15]   # §15 panels 3/4 (Q̇≠0): VII/VIII + A'/B' (2N² extra HVPs)
                    sec15p34 = _sec15_panel34(hvp_at15, tm2["th"], Jg15, tm1["J"], tm2["J"], tm1["r"], tm2["r"], u1_15, Amat15, Bmat15, sec15, lr, N, diveps=P.get("divreg", 0.3))
                sec15_hist.append({"th": th15.detach().clone(), "J": Jg15.detach(), "r": r15.detach(), "t": t})
                if len(sec15_hist) > 2:
                    sec15_hist.pop(0)
                if f64_15:                                # advance the float64 shadow one GD step (same rule/minibatch as the displayed
                    sec15_th64 = sec15_th64 - lr * _opt_dir(   # float32 trajectory, but in float64 ⇒ next step's D² stays cancellation-free)
                        _TL.model, gradL(sec15_th64, X15, Y.double())[0], opt)

            # report σ₁ per-sample (÷N) so the theory matches the true sharpness λmax(∇²L) ≈ λmax(GN) = σ₁/N
            thPr = [(x / N if x is not None else None) for x in thP] if thP else None
            thAr = [(x / N if x is not None else None) for x in thA] if thA else None
            thPpsdR = [(x / N if x is not None else None) for x in thPpsd] if thPpsd else None
            thPdR = [(x / N if x is not None else None) for x in thP_d] if thP_d else None
            thPpsddR = [(x / N if x is not None else None) for x in thPpsd_d] if thPpsd_d else None

            yield {
                "type": "step", "t": t, "steps": steps, "p": p,
                "loss": loss, "sharp": sharp,
                "thP": thPr, "thA": thAr, "thPpsd": thPpsdR, "thAH": thAH,
                "thP_d": thPdR, "thPpsd_d": thPpsddR,
                **cub,    # §10 cubic keys: c47/c47p, c51/c51p, c51n/c51np, cActN, cActH, cdQ, cdJ (when s17)
                "r": ([] if r is None else r[:nResid].detach().cpu().tolist()),
                "hfMax": (feTop[0] / N if feTop else None),         # ÷N: function-Hessian H=∇²Σf is summed over the
                "hfMin": (feBot[0] / N if feBot else None),         #   N samples → divide to match the per-sample loss
                "hfTop": [v / N for v in feTop[:n]], "hfBot": [v / N for v in feBot[:n]],   # Hessian (G+S) scale (mirrors eos_lab)
                "hlTop": hlt, "hlBot": hlb,
                "gnTop": gnt, "gnBot": gnb, "srTop": srt, "srBot": srb,
                "jt": jt, "jb": jb, "jtN": jtN, "jbN": jbN,
                "paPos": paPos, "paNeg": paNeg, "dimPos": dimPos, "dimNeg": dimNeg,
                "ntkR": ntkR, "ntkH": ntkH, "ntkGs": ntkGs, "ntkGsA": ntkGsA, "ntkGl": ntkGl, "ntkGlA": ntkGlA,
                "fhEvT": fhEvT, "fhEvB": fhEvB,
                "jhe1": jhe1, "jhe2": jhe2, "jhe1b": jhe1b, "jhe2b": jhe2b,
                "jh2e1": jh2e1, "jh2e2": jh2e2, "jh2e1b": jh2e1b, "jh2e2b": jh2e2b,
                "g2J": g2J, "g2Jn": g2Jn,
                "q9t": q9t, "q9b": q9b, "q9tN": q9tN, "q9bN": q9bN,
                "q9tR": q9tR, "q9bR": q9bR, "q9gt": q9gt, "q9gb": q9gb,
                "q10": q10, "q10N": q10N,
                "d4Ht": d4Ht, "d4Hb": d4Hb, "d4Qt": d4Qt, "d4Qb": d4Qb,
                "sps": (t - start + 1) / max(time.time() - t0, 1e-9),
            }

            if s5 and eigTick % slqStride == 0:
                ntr = max(nProbe, 4)   # Hutchinson trace probes (a few extra for a smoother trace curve)
                yield {
                    "type": "slq", "t": t,
                    "sH": slq_density(lambda v: hvpF(th, X, v), p, nProbe, mSLQ, 80, 0x11),
                    "sG": slq_density(lambda v: hvpG(th, X, v), p, nProbe, mSLQ, 80, 0x22),
                    "sS": slq_density(lambda v: hvpS(th, X, v, cS), p, nProbe, mSLQ, 80, 0x33),
                    "sHL": slq_density(lambda v: hvpL(th, X, Y, v), p, nProbe, mSLQ, 80, 0x44),
                    # trace of each operator vs iteration (Hutchinson). H is ÷N → per-sample scale (matches §1).
                    "trH": _hutch_trace(lambda v: hvpF(th, X, v), p, ntr, 0x51) / N,
                    "trG": _hutch_trace(lambda v: hvpG(th, X, v), p, ntr, 0x52),
                    "trS": _hutch_trace(lambda v: hvpS(th, X, v, cS), p, ntr, 0x53),
                    "trHL": _hutch_trace(lambda v: hvpL(th, X, Y, v), p, ntr, 0x54),
                }
            if g3d is not None:
                yield {"type": "g3d", "t": t, **g3d}     # §11 3D-grid snapshot (browser stores + scrubs these)
            if sec12 is not None:
                yield {"type": "g4d", "t": t, **sec12}   # §12 Hessian-eigvec cross-similarity snapshot
            if sec13 is not None:
                yield {"type": "g13", "t": t, **sec13}   # §13 G1/G2/G3 approximations vs exact J_iᵀQ_jJ_k·r_k
            if sec14 is not None:
                yield {"type": "g14", "t": t, **sec14}   # §14 Tr(ΔNTK) per-triplet decomposition snapshot
            if sec15 is not None:
                yield {"type": "g15", "t": t, **sec15}   # §15 2nd-difference decomposition of ‖J‖²_F & σ₁
            if sec15n is not None:
                yield {"type": "g15n", "t": t, **sec15n}  # §15 panel 5 — per-sample norm evolution (mean ± std)
            if sec15b is not None:
                yield {"type": "g15b", "t": t, **sec15b}  # §15 new Panel B — per-sample top/bottom projection ratio (mean ± std)
            if sec15p34 is not None:
                yield {"type": "g15p34", "t": t, **sec15p34}  # §15 panels 3/4 — Q̇≠0 terms VII/VIII + A'/B'
            eigTick += 1
            done += 1

        th = th - lr * _opt_dir(_TL.model, gradL(th, X, Y)[0], opt)

    yield {"type": "done", "p": p}


def _resid_hist(rA, rS, bins=40):
    """Overlaid histograms of the d_n=N·d_out residual elements: actual vs surrogate, shared bins."""
    a = rA.detach().to("cpu", torch.float64)
    s = rS.detach().to("cpu", torch.float64)
    lo = float(min(a.min(), s.min()))
    hi = float(max(a.max(), s.max()))
    if not hi > lo:
        hi = lo + 1.0
    edges = torch.linspace(lo, hi, bins + 1)
    centers = (0.5 * (edges[:-1] + edges[1:])).tolist()
    ha = torch.histc(a, bins=bins, min=lo, max=hi).tolist()
    hs = torch.histc(s, bins=bins, min=lo, max=hi).tolist()
    return {"type": "hist", "x": centers, "ya": ha, "ys": hs}


# ===================== surrogate-comparison run (quadratic / linear approximation) =====================
def run_surrogate_compare(P):
    """Train the actual model AND a frozen-window surrogate in lockstep and stream their comparison.
    surrogate='quad': 2nd-order Taylor f₀+J₀Δθ+½ΔθᵀQΔθ; 'linear': f₀+J₀Δθ (drop curvature).
    Q (function Hessian) and J₀ are frozen at each window start (every `qapprox` steps); the surrogate
    runs GD on the loss of that approximation. Loss = MSE or (small-class) cross-entropy — everything
    loss-touching goes through the loss object, so the surrogate GD uses ∂L/∂out and the surrogate loss
    Hessian uses the matching Gauss–Newton metric. The §9 σ₁ theory is squared-loss only (skipped for CE).
    Panels: loss, residual mean/std, top-n loss-Hessian eigenvalues, residual histogram, §9 theory σ₁,
    §4 and §4d projections."""
    kind = P.get("surrogate", "quad")
    dataset = P.get("dataset", "synthetic")
    arch = P.get("arch", "mlp")
    if dataset == "owt":
        # The quadratic/linear surrogate freezes the FULL per-output Jacobian J₀ (and, in the multi-sample
        # σ₁ theory, an M×M NTK Gram) with M = nsamp·seqlen·vocab rows. At GPT-2's 50257-token vocab that
        # is ~10⁸–10¹¹ rows — forming J₀ (M×p) or the M×M Gram is many orders of magnitude beyond memory.
        # This is a hard scaling limit of the surrogate construction, independent of the loss. (MSE on the
        # LM IS now supported in the MAIN section; the Taylor σ₁ theory there is also gated off for OWT by
        # the same multi_ok size budget.) So OpenWebText has no surrogate comparison at realistic vocab.
        reason = ("the Taylor σ₁ theory is squared-loss only AND " if P.get("loss") == "ce" else "")
        yield {"type": "error", "error": "OpenWebText has no quadratic/linear surrogate comparison: " + reason +
               "the surrogate must materialise the full per-output Jacobian J₀ (M = N·seqlen·vocab rows; "
               "≈10⁸–10¹¹ at GPT-2's 50257 vocab) and an M×M NTK Gram — far beyond memory. This is a hard "
               "size limit, not a toggle. Run OpenWebText from the main section above (§1/§2/§3/§5 + §4/§6), "
               "which fully supports both CE and MSE."}
        return
    if dataset == "cifar10":
        inDimE, outDimE = 3072, 10
    elif dataset == "cifar2":
        inDimE, outDimE = 3072, 1                         # 2-class CIFAR cast as SCALAR regression (±1)
    elif dataset == "mnist":
        inDimE, outDimE = 3072, 10                        # MNIST padded to 3×32×32 (drop-in for cifar-shaped nets)
    elif dataset == "mnist2":
        inDimE, outDimE = 3072, 1                         # 2-class MNIST cast as SCALAR regression (±1)
    elif dataset == "sorting":
        inDimE = outDimE = int(P["seqlen"])
    elif dataset == "chebyshev":
        inDimE = outDimE = 1                              # scalar x → scalar T_k(x)
    else:
        inDimE, outDimE = P["indim"], P["outdim"]
    _TL.model = build_model(arch, inDimE, outDimE, P)
    _TL.loss = build_loss(P.get("loss", "mse"))           # MSE, or small-class cross-entropy
    ceLoss = _TL.loss.name == "ce"
    p = _TL.model.p
    N, inD, outD = P["nsamp"], inDimE, outDimE
    M = N * outD
    multi = M > 1
    half = max(1, p // 2)
    n = min(max(1, P["neig"]), half)
    lr = P["lr"]
    opt = P.get("optimizer", "gd")
    thr = 2.0 / lr
    ee = max(1, P["eigevery"])
    steps = P["steps"]
    W = max(1, P["qapprox"])
    nProbe = max(1, P["slqprobes"])
    qmode = P["qmode"]
    tset = P["tset"]
    mV = min(p, max(2 * n + 16, 28))
    cL, cR, cE, cH, cT, c4, c4d = (P["c1"], P["c2"], P["c3"], P["c4"], P["c5"], P["c6"], P["c7"])
    c7a = P.get("c8", True)         # §7a NTK alignment panel (actual model)
    c9c = P.get("c9", False)        # §9c: σ₁ predictions vs the full loss-Hessian sharpness λmax(∇²L)
    c9d = P.get("c10", False)       # §9d: predictions with the residual self-computed by the quadratic model
    c9dc = P.get("c11", False)      # §9d-c: §9d predictions vs the full loss-Hessian sharpness
    cT = cT and not ceLoss          # the Eq-13/21/22/23/29 σ₁ theory (§9) is derived for squared loss only
    c7a = c7a and not ceLoss        # NTK alignment uses the MSE residual r → MSE only (multi-class CE: off)
    c9c = c9c and not ceLoss        # §9c is the same squared-loss σ₁ recursion → MSE only
    c9d = c9d and not ceLoss; c9dc = c9dc and not ceLoss   # §9d/§9d-c likewise squared-loss only (unreachable: surrogate is always MSE)
    cCub = P.get("c12", False) and not ceLoss   # §10 cubic approximation (actual model) vs the surrogate σ₁ + actual λmax
    cubCtx = {"N": N, "p": p, "M": M, "lr": lr, "ee": ee, "cubicapprox": max(1, P.get("cubicapprox", 10)),
              "cn": max(1, min(nProbe, 4)), "multi": multi, "multi_ok": multi}
    cubSt = cubic_init_state()

    if p > PMAX:
        yield {"type": "meta", "error": f"p={p} parameters exceeds the cap ({PMAX}). Reduce the model size."}
        return
    yield {"type": "meta", "p": p, "n": n, "thr": thr, "kind": kind, "multi": multi, "M": M,
           "device": f"{_dev()} · {str(DTYPE).replace('torch.', '')}"}

    th, X, Y, pos_rows, neg_rows = init_data_theta(P, dataset, N, inD, outD)
    Yf = Y.reshape(-1)
    zero_p = torch.zeros(p, dtype=DTYPE, device=_dev())

    mytok = P.get("_token", 0)                          # per-device token claimed in _sse via acquire_device()
    _devkey = str(_dev())
    start = max(0, min(int(P.get("start", 0)), steps))
    slqStride = max(1, math.ceil((steps // ee + 1) / 50))
    heavyevery = max(1, P.get("heavyevery", 4))   # §7-proj + §8 throttle (mirror run_stream)
    t0 = time.time()
    eigTick = 0
    # frozen-window state
    th0 = J0 = gF0 = f0flat = th_sur = feTV0 = feBV0 = None
    thBase = 0.0
    thJ = thFroz = thJp = None
    thAcc1 = thAcc2 = 0.0
    thProd3 = thProd4 = thProd5 = 1.0
    thAccPSD = thAccPSD1 = 0.0      # §9b: accumulated 2nd-order PSD term ‖ΔJᵀu₁‖² (≥0), multi / single
    # §9d: identical predictions, residual SELF-COMPUTED by the frozen quadratic model (reuses J0/f0flat)
    thDth = None; thJ_d = None; thProd3_d = 1.0; thProd4_d = 1.0; thProd5_d = 1.0; thAcc2_d = 0.0; thAccPSD_d = 0.0
    thDth_s = None; thJp_d = None; thAcc1_d = 0.0; thAccPSD1_d = 0.0
    # §7a Δλ·Δr running products (sync + 1-step lagged), of the ACTUAL model
    prevSharp7a = prevNtkR7a = prevPrevNtkR7a = None
    sumGsync7a = sumGlag7a = None; cntGs7a = cntGl7a = 0

    for t in range(steps + 1):
        if mytok != RUN_TOKEN.get(_devkey, mytok):
            return
        if t == start:
            t0 = time.time()
        if t % W == 0:                                       # window start: freeze θ₀, J₀, Q; reset surrogate
            th0 = th.clone()
            J0, f0flat = jac_cols(th0, X)
            gF0 = gradF(th0, X)[0]
            th_sur = th0.clone()
            if c4 or c4d:
                feTV0, feBV0, _, _ = lanczos_extreme(lambda v: hvpF(th0, X, v), p, n, mV, 0xF00D)
                feTV0 = [pin_sign(x) for x in feTV0]
                feBV0 = [pin_sign(x) for x in feBV0]
            thAcc1 = thAcc2 = 0.0
            thProd3 = thProd4 = thProd5 = 1.0
            thAccPSD = thAccPSD1 = 0.0
            if cT or c9c or c9d or c9dc:
                if multi:
                    thJ = J0.clone()
                    thFroz = fh_frozen(th0, X, nProbe, min(n, M), 2, mV, M)
                    thBase = float(_safe_eigvalsh(thJ @ thJ.t())[-1])
                    if c9d or c9dc:   # §9d: reset quad-GD displacement & parallel accumulators (J0/f0flat already frozen above)
                        thDth = torch.zeros(p, dtype=DTYPE, device=_dev()); thJ_d = J0.clone()
                        thProd3_d = 1.0; thProd4_d = 1.0; thProd5_d = 1.0; thAcc2_d = 0.0; thAccPSD_d = 0.0
                    else:
                        thJ_d = None
                else:
                    thJp = gF0.clone()
                    thBase = float(gF0 @ gF0)
                    if c9d or c9dc:
                        thDth_s = torch.zeros(p, dtype=DTYPE, device=_dev()); thJp_d = gF0.clone()
                    else:
                        thJp_d = None
                    thAcc1_d = 0.0; thAccPSD1_d = 0.0

        if t >= start and t % ee == 0:
            # ---- actual model ----
            J_act, out_act = gradF(th, X)
            lossA = float(_TL.loss.value(out_act, Y, N))
            if not math.isfinite(lossA) or lossA > 1e10:
                yield {"type": "error", "error": f"actual run diverged at step {t} (loss={lossA:.2e})."}
                return
            # "residual" = −N·∂L/∂out: Y−f for MSE, Y−softmax(f) for CE (loss-generic via the cotangent)
            rA = (-N * _TL.loss.resid_cotangent(out_act, Y, N)).reshape(-1)
            # ---- surrogate forward / gradient (frozen Q,J₀ at θ₀) ----
            dth = th_sur - th0
            f_sur = f0flat + (J0 @ dth)
            Jq = J0
            if kind == "quad":
                QD = jac_hvp(th0, X, dth)
                f_sur = f_sur + 0.5 * (QD @ dth)
                Jq = J0 + QD
            elif kind == "cubic":     # f₀+J₀Δθ+½ΔθᵀQΔθ+⅙T[Δθ,Δθ,Δθ];  Jacobian J₀+Q[Δθ]+½T[Δθ,Δθ]
                QD = jac_hvp(th0, X, dth)
                TDD = sur_T2(th0, X, dth)
                f_sur = f_sur + 0.5 * (QD @ dth) + (1.0 / 6.0) * (TDD @ dth)
                Jq = J0 + QD + 0.5 * TDD
            out_sur = f_sur.reshape(N, outD)
            rS = (-N * _TL.loss.resid_cotangent(out_sur, Y, N)).reshape(-1)
            lossS = float(_TL.loss.value(out_sur, Y, N))

            rmeanA, rstdA = float(rA.mean()), float(rA.std(unbiased=False))
            rmeanS, rstdS = float(rS.mean()), float(rS.std(unbiased=False))

            # ---- top-n loss-Hessian eigenvalues (actual & surrogate) ----
            eigA = eigS = None
            sharpA = sharpS = None
            if cE:
                eigA, _ = lanczos_extreme_vals(lambda v: hvpL(th, X, Y, v), p, n, mV, 0x5EED1)
                sharpA = eigA[0]
                # surrogate loss Hessian = Gauss-Newton(Jq) + residual-curvature with the curvature FROZEN at θ₀.
                # Both quad and linear include the frozen residual term so that at a window start (Jq=J₀,
                # f_sur=f₀) the surrogate Hessian equals the actual loss Hessian G+S and the eigenvalues snap.
                # surrogate Gauss–Newton uses the loss-aware output metric A (=I/N for MSE, the softmax
                # GGN for CE); the residual-curvature term keeps the FROZEN cotangent at θ₀.
                csur = _TL.loss.resid_cotangent(out_sur, Y, N)
                surHL = lambda v: (Jq.t() @ _TL.loss.gn_apply(out_sur, (Jq @ v).reshape(N, outD), N).reshape(-1)
                                   + hvpS(th0, X, v, csur))
                eigS, _ = lanczos_extreme_vals(surHL, p, n, mV, 0x5EED2)
                sharpS = eigS[0]
            if (c9c or c9dc) and sharpA is None:   # §9c/§9d-c need the actual model's λmax(∇²L) even if §eig (cE) is off
                sharpA = float(lanczos_extreme_vals(lambda v: hvpL(th, X, Y, v), p, 1, mV, 0x5EED1)[0][0])

            # ---- §4 projections: J onto top/bottom-n eigvecs of H (actual own H, surrogate frozen H₀) ----
            feTVa = feBVa = None
            j4tA = j4bA = j4tS = j4bS = None
            if (c4 or c4d) and feTV0 is not None:
                # use the SAME Lanczos seed as the frozen H₀ eigvecs (feTV0, 0xF00D): at a window start
                # θ=θ₀ so the operator and probe are identical → feTVa==feTV0 and §4/§4d snap exactly.
                feTVa, feBVa, _, _ = lanczos_extreme(lambda v: hvpF(th, X, v), p, n, mV, 0xF00D)
                feTVa = [pin_sign(x) for x in feTVa]
                feBVa = [pin_sign(x) for x in feBVa]
            if c4 and feTV0 is not None:
                J_sur = gF0 + (hvpF(th0, X, dth) if kind in ("quad", "cubic") else zero_p)
                if kind == "cubic":
                    J_sur = J_sur + 0.5 * sur_T2(th0, X, dth).sum(0)   # + ½ Σ_a T₀_a[Δθ,Δθ] (cubic ∇Σf surrogate)
                j4tA = [float(J_act @ feTVa[k]) for k in range(n)]
                j4bA = [float(J_act @ feBVa[k]) for k in range(n)]
                j4tS = [float(J_sur @ feTV0[k]) for k in range(n)]
                j4bS = [float(J_sur @ feBV0[k]) for k in range(n)]

            # ---- §4d projections: per-group J (±residual-sign groups) onto H eigvecs ----
            d4tA = d4bA = d4tS = d4bS = None
            if c4d and feTV0 is not None:
                Jc_act, _ = jac_cols(th, X)

                def grp(Jm, rows):
                    return Jm[rows].sum(0) if rows.numel() else zero_p
                JpA, JnA = grp(Jc_act, pos_rows), grp(Jc_act, neg_rows)
                JpS, JnS = grp(Jq, pos_rows), grp(Jq, neg_rows)
                d4tA = [float(JpA @ feTVa[k]) for k in range(n)] + [float(JnA @ feTVa[k]) for k in range(n)]
                d4bA = [float(JpA @ feBVa[k]) for k in range(n)] + [float(JnA @ feBVa[k]) for k in range(n)]
                d4tS = [float(JpS @ feTV0[k]) for k in range(n)] + [float(JnS @ feTV0[k]) for k in range(n)]
                d4bS = [float(JpS @ feBV0[k]) for k in range(n)] + [float(JnS @ feBV0[k]) for k in range(n)]

            # ---- §7a NTK alignment (actual model): residual→NTK eigvec; NTK eigvec→FH right-sing vec ----
            ntkR = ntkH = None
            ntkGs = ntkGsA = ntkGl = ntkGlA = None      # §7a Δλ·Δr products (sync + lagged) and running avgs
            if c7a and multi:
                Jc_a, _ = jac_cols(th, X)
                Kv7, Vk7 = sym_eig_desc(Jc_a @ Jc_a.t())
                for k in range(min(n, M)):
                    Vk7[:, k] = pin_sign(Vk7[:, k])
                GH7 = torch.zeros(M, M, dtype=DTYPE, device=_dev())
                for pr in range(nProbe):
                    Hz = jac_hvp(th, X, _randn_vec(p, (0xC0FFEE + pr * 40503) & 0xFFFFFFFF))
                    GH7 += Hz @ Hz.t()
                GH7 /= nProbe
                _, Vh7 = sym_eig_desc(GH7)
                for k in range(min(n, M)):
                    Vh7[:, k] = pin_sign(Vh7[:, k])
                nntk = min(n, M)
                ntkR = [float(rA @ Vk7[:, k]) for k in range(nntk)]
                ntkH = [float(Vk7[:, 0] @ Vh7[:, k]) for k in range(nntk)]
                # §7a products: Δsharp(t)·Δ⟨r,vₖ⟩ — synchronous and 1-step lagged — each with a running time-avg.
                if sumGsync7a is None:
                    sumGsync7a = [0.0] * nntk; sumGlag7a = [0.0] * nntk
                dSh = (sharpA - prevSharp7a) if (prevSharp7a is not None and sharpA is not None) else None
                if dSh is not None and prevNtkR7a is not None:
                    cntGs7a += 1; ntkGs = []; ntkGsA = []
                    for k in range(nntk):
                        g = dSh * (ntkR[k] - prevNtkR7a[k]); sumGsync7a[k] += g
                        ntkGs.append(g); ntkGsA.append(sumGsync7a[k] / cntGs7a)
                if dSh is not None and prevPrevNtkR7a is not None:
                    cntGl7a += 1; ntkGl = []; ntkGlA = []
                    for k in range(nntk):
                        g = dSh * (prevNtkR7a[k] - prevPrevNtkR7a[k]); sumGlag7a[k] += g
                        ntkGl.append(g); ntkGlA.append(sumGlag7a[k] / cntGl7a)
                prevPrevNtkR7a = prevNtkR7a; prevNtkR7a = list(ntkR); prevSharp7a = sharpA

            # ---- §9 theory σ₁ (unchanged frozen-Q propagation) vs the SURROGATE's measured σ₁ ----
            # §9b adds the dropped 2nd-order PSD term; §9c compares against the actual model's full
            # loss-Hessian sharpness λmax(∇²L) (=sharpA) instead of the surrogate's Gauss-Newton edge.
            thP = thA = thPpsd = None
            thP_d = thPpsd_d = None       # §9d / §9d-c: predictions with the quad-self-computed residual
            thAH = sharpA if (c9c or c9dc) else None
            if cT or c9c or c9d or c9dc:
                etaN = lr / max(N, 1)
                reps_ = max(1, ee)
                thP = [None]*5      # display cols 1-5: Eq-13, Eq-21, Eq-22, Eq-23, Eq-29 (col-4=thProd5=Eq-23, col-5=thProd4=Eq-29)
                thA = [None]*5
                thPpsd = [None]*5
                if multi and thJ is not None and thFroz is not None:
                    rr = rA
                    Jt = thJ
                    Fz = thFroz
                    Kw, Vw = sym_eig_desc(Jt @ Jt.t())
                    NV0 = min(n, M)
                    for k in range(NV0):
                        Vw[:, k] = pin_sign(Vw[:, k])
                    sig1 = max(float(Kw[0]), 1e-30)
                    sgT = math.sqrt(sig1)
                    u1 = Vw[:, 0]
                    v1 = Jt.t() @ u1
                    v1 = v1 / max(float(v1.norm()), 1e-30)
                    sigSur = float(_safe_eigvalsh(Jq @ Jq.t())[-1])   # surrogate σ₁ (measured)
                    thA[1] = thA[2] = thA[3] = thA[4] = sigSur
                    cap = 10 * max(sigSur, 1e-30)
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
                    thP[1] = clmp(thBase + thAcc2)
                    thP[2] = clmp(thBase * thProd3)
                    thP[3] = clmp(thBase * thProd5)                       # col-4 = Eq-23 (swapped)
                    thP[4] = clmp(thBase * thProd4)                       # col-5 = Eq-29 (swapped)
                    thPpsd[1] = clmp(thBase + thAcc2 + thAccPSD)          # §9b: + 2nd-order PSD term Σ‖ΔJᵀu₁‖²
                    thPpsd[2] = clmp(thBase * thProd3 + thAccPSD)
                    thPpsd[3] = clmp(thBase * thProd5 + thAccPSD)         # col-4 = Eq-23
                    thPpsd[4] = clmp(thBase * thProd4 + thAccPSD)         # col-5 = Eq-29
                    qv1 = hvpS(th0, X, v1, u1.reshape(N, outD))        # Q[u₁]v₁ via the weighted HVP (2 grad evals, not the full
                    #   M×p jac_hvp) ⇒ the EXACT bilinear v₁ᵀQ[u₁]v_k = qv1·v_k (used by Eq-22 & Eq-23)
                    pproj = float(rr @ u1)                            # col-3 (Eq-22): p_t = r·u₁ (the v₁-coeff of Jᵀr)
                    thProd3 *= (1 + 2 * etaN * pproj * float(qv1 @ v1)) ** reps_   # σ_{t+1}=σ_t[1+2η (r·u₁) v₁ᵀQ[u₁]v₁]  (exact)
                    S4 = 0.0
                    S5 = 0.0
                    NV = min(max(tset, 1), NV0)
                    for vk in range(NV):
                        sgv = math.sqrt(max(float(Kw[vk]), 1e-30))
                        rho = float(rr @ Vw[:, vk]); gk = gnv(vk)
                        S4 += (sgv / sgT) * fhBil(gk) * rho            # col-5 (Eq-29): bilinear via the γτz eigendecomposition of Q (approx)
                        S5 += (sgv / sgT) * float(qv1 @ gk) * rho      # col-4 (Eq-23): bilinear v₁ᵀQ[u₁]v_k computed directly (exact)
                    thProd4 *= (1 + 2 * etaN * S4) ** reps_             # col-5 (Eq-29)
                    thProd5 *= (1 + 2 * etaN * S5) ** reps_            # col-4 (Eq-23): differs from Eq-29 only by the Q-decomposition approx
                    for _ in range(reps_):
                        QW = jac_hvp(th0, X, thJ.t() @ rr)
                        qu = (u1.unsqueeze(1) * QW).sum(0)
                        thAcc2 += 2 * etaN * sgT * float(v1 @ qu)
                        thAccPSD += (etaN ** 2) * float(qu @ qu)         # §9b: ‖ΔJᵀu₁‖² = (η/N)²‖qu‖²
                        thJ = thJ + etaN * QW
                elif not multi and thJp is not None:
                    sigSur = float(Jq.reshape(-1) @ Jq.reshape(-1))
                    thA[0] = sigSur
                    cap0 = 10 * max(sigSur, 1e-30)
                    thP[0] = min(max(thBase + thAcc1, 0.0), cap0)
                    thPpsd[0] = min(max(thBase + thAcc1 + thAccPSD1, 0.0), cap0)   # §9b: + Σ‖ΔJ‖²
                    rsc = float((Yf - out_act.reshape(-1))[0])
                    for _ in range(reps_):
                        QJ = hvpF(th0, X, thJp)
                        thAcc1 += 2 * lr * rsc * float(thJp @ QJ)
                        thAccPSD1 += ((lr * rsc) ** 2) * float(QJ @ QJ)   # §9b: ‖ΔJ‖² = (η r)²‖QJ‖²
                        thJp = thJp + lr * rsc * QJ

                # ---- §9d: SAME predictions, residual SELF-COMPUTED by the frozen quadratic model (closed loop) ----
                thP_d = [None]*5; thPpsd_d = [None]*5
                if multi and thJ_d is not None and thFroz is not None and J0 is not None:
                    Jd = thJ_d; Fz = thFroz; dth = thDth
                    HzD = jac_hvp(th0, X, dth)                            # Q[Δθ] = {∇²f_a(θ₀)·Δθ}, (M,p)
                    r_q = Yf - (f0flat.reshape(-1) + J0 @ dth + 0.5 * (HzD @ dth))   # quad-model residual
                    Kw, Vw = sym_eig_desc(Jd @ Jd.t())
                    NV0 = min(n, M)
                    for k in range(NV0):
                        Vw[:, k] = pin_sign(Vw[:, k])
                    sgT = math.sqrt(max(float(Kw[0]), 1e-30)); u1 = Vw[:, 0]
                    v1 = Jd.t() @ u1; v1 = v1 / max(float(v1.norm()), 1e-30)
                    cap = 10 * max(float(sigSur), 1e-30); clmpd = lambda x: min(max(x, 0.0), cap)
                    def gnv_d(k):
                        g = Jd.t() @ Vw[:, k]; return g / max(float(g.norm()), 1e-30)
                    def fhBil_d(vk):
                        s = 0.0
                        for i in range(len(Fz["gamma"])):
                            yu = float(Fz["y"][i] @ u1)
                            sj = sum(Fz["tau"][i][j]*float(v1 @ Fz["z"][i][j])*float(vk @ Fz["z"][i][j]) for j in range(len(Fz["z"][i])))
                            s += Fz["gamma"][i]*sj*yu
                        return s
                    thP_d[1] = clmpd(thBase + thAcc2_d)
                    thP_d[2] = clmpd(thBase * thProd3_d); thP_d[3] = clmpd(thBase * thProd5_d); thP_d[4] = clmpd(thBase * thProd4_d)
                    thPpsd_d[1] = clmpd(thBase + thAcc2_d + thAccPSD_d)
                    thPpsd_d[2] = clmpd(thBase * thProd3_d + thAccPSD_d); thPpsd_d[3] = clmpd(thBase * thProd5_d + thAccPSD_d)
                    thPpsd_d[4] = clmpd(thBase * thProd4_d + thAccPSD_d)
                    qv1 = hvpS(th0, X, v1, u1.reshape(N, outD))
                    pproj = float(r_q @ u1)
                    thProd3_d *= (1 + 2 * etaN * pproj * float(qv1 @ v1)) ** reps_
                    S4 = 0.0; S5 = 0.0; NV = min(max(tset, 1), NV0)
                    for vk in range(NV):
                        sgv = math.sqrt(max(float(Kw[vk]), 1e-30)); rho = float(r_q @ Vw[:, vk]); gk = gnv_d(vk)
                        S4 += (sgv/sgT) * fhBil_d(gk) * rho
                        S5 += (sgv/sgT) * float(qv1 @ gk) * rho
                    thProd4_d *= (1 + 2 * etaN * S4) ** reps_
                    thProd5_d *= (1 + 2 * etaN * S5) ** reps_
                    for _ in range(reps_):                                # advance quad GD (Eq-15 with r_q) + Δθ
                        g = thJ_d.t() @ r_q
                        QW = jac_hvp(th0, X, g)
                        qu = (u1.unsqueeze(1) * QW).sum(0)
                        thAcc2_d += 2 * etaN * sgT * float(v1 @ qu)
                        thAccPSD_d += (etaN ** 2) * float(qu @ qu)
                        thJ_d = thJ_d + etaN * QW
                        thDth = thDth + etaN * g
                elif (not multi) and thJp_d is not None:
                    Qd = hvpF(th0, X, thDth_s)                            # ∇²f(θ₀)·Δθ
                    r_qs = float(Yf[0] - (float(f0flat.reshape(-1)[0]) + float(gF0 @ thDth_s) + 0.5 * float(thDth_s @ Qd)))
                    cap0 = 10 * max(float(thJp_d @ thJp_d), 1e-30)
                    thP_d[0] = min(max(thBase + thAcc1_d, 0.0), cap0)
                    thPpsd_d[0] = min(max(thBase + thAcc1_d + thAccPSD1_d, 0.0), cap0)
                    for _ in range(reps_):
                        QJ = hvpF(th0, X, thJp_d)
                        thAcc1_d += 2 * lr * r_qs * float(thJp_d @ QJ)
                        thAccPSD1_d += ((lr * r_qs) ** 2) * float(QJ @ QJ)
                        thJp_d = thJp_d + lr * r_qs * QJ
                        thDth_s = thDth_s + lr * r_qs * thJp_d

            # ---- §10 CUBIC approximation (actual model) — predictions vs the SURROGATE's σ₁ + actual λmax ----
            cub = {}
            if cCub:
                shCub = sharpA if sharpA is not None else float(
                    lanczos_extreme_vals(lambda v: hvpL(th, X, Y, v), p, 1, mV, 0x5EED1)[0][0])
                sigSurC = (float(_safe_eigvalsh(Jq @ Jq.t())[-1]) if multi
                           else float(Jq.reshape(-1) @ Jq.reshape(-1)))   # surrogate σ₁ (the §10 reference here)
                cub = cubic_step(cubCtx, cubSt, th, X, Y, t, J_act, out_act, rA, [sigSurC], shCub)
                if "cActN" in cub:
                    cub["cActN"] = sigSurC / N      # compare the cubic prediction to the SURROGATE's σ₁ (per-sample)

            # report σ₁ per-sample (÷N) so the theory matches the true sharpness scale (σ₁/N)
            thPr = [(x / N if x is not None else None) for x in thP] if thP else None
            thAr = [(x / N if x is not None else None) for x in thA] if thA else None
            thPpsdR = [(x / N if x is not None else None) for x in thPpsd] if thPpsd else None
            thPdR = [(x / N if x is not None else None) for x in thP_d] if thP_d else None
            thPpsddR = [(x / N if x is not None else None) for x in thPpsd_d] if thPpsd_d else None

            yield {
                "type": "step", "t": t, "steps": steps, "p": p, "kind": kind,
                **cub,    # §10 cubic keys (when c12): c47/c47p, c51/c51p, c51n/c51np, cActN(=surrogate σ₁), cActH, cdQ, cdJ
                "lossA": lossA if cL else None, "lossS": lossS if cL else None,
                "rmeanA": rmeanA if cR else None, "rstdA": rstdA if cR else None,
                "rmeanS": rmeanS if cR else None, "rstdS": rstdS if cR else None,
                "eigA": eigA, "eigS": eigS, "sharpA": sharpA, "sharpS": sharpS,
                "thP": thPr, "thA": thAr, "thPpsd": thPpsdR, "thAH": thAH,
                "thP_d": thPdR, "thPpsd_d": thPpsddR,
                "j4tA": j4tA, "j4bA": j4bA, "j4tS": j4tS, "j4bS": j4bS,
                "d4tA": d4tA, "d4bA": d4bA, "d4tS": d4tS, "d4bS": d4bS,
                "ntkR": ntkR, "ntkH": ntkH, "ntkGs": ntkGs, "ntkGsA": ntkGsA, "ntkGl": ntkGl, "ntkGlA": ntkGlA,
                "sps": (t - start + 1) / max(time.time() - t0, 1e-9),
            }
            if cH and eigTick % slqStride == 0:
                yield _resid_hist(rA, rS)
            eigTick += 1

        # advance the actual model AND the frozen-window surrogate by one GD step EVERY iteration, so the
        # surrogate stays in lockstep with the actual model regardless of eigevery / qapprox alignment
        # (the surrogate GD is on the frozen-θ₀ MSE Taylor model; Q,J₀ are frozen at the window start).
        th = th - lr * _opt_dir(_TL.model, gradL(th, X, Y)[0], opt)   # actual trajectory uses the chosen optimizer
        if th0 is not None:
            dths = th_sur - th0
            if kind == "quad":
                QDs = jac_hvp(th0, X, dths)
                fss = f0flat + (J0 @ dths) + 0.5 * (QDs @ dths)
                Jqs = J0 + QDs
            elif kind == "cubic":
                QDs = jac_hvp(th0, X, dths)
                TDDs = sur_T2(th0, X, dths)
                fss = f0flat + (J0 @ dths) + 0.5 * (QDs @ dths) + (1.0 / 6.0) * (TDDs @ dths)
                Jqs = J0 + QDs + 0.5 * TDDs
            else:
                fss = f0flat + (J0 @ dths)
                Jqs = J0
            gss = _TL.loss.resid_cotangent(fss.reshape(N, outD), Y, N).reshape(-1)   # MSE: (fss−Y)/N
            th_sur = th_sur - lr * (Jqs.t() @ gss)

    yield {"type": "done", "p": p}


# ===================== HTTP / SSE =====================
def _parse_params(q):
    def g(k, d):
        return q.get(k, [d])[0]

    def fi(k, d):
        return int(float(g(k, d)))

    def ff(k, d):
        return float(g(k, d))

    return {
        "depth": fi("depth", 5), "width": fi("width", 1), "act": g("act", "linear"),
        "bias": g("bias", "0"), "fixedx": g("fixedx", "1"),
        "lr": ff("lr", 0.36), "init": ff("init", 0.4),
        "nsamp": fi("nsamp", 1), "batch": fi("batch", 0), "indim": fi("indim", 1), "outdim": fi("outdim", 1),
        "tgt": ff("tgt", 1.0), "neig": fi("neig", 3), "kth": fi("kth", 1),
        "steps": fi("steps", 400), "eigevery": fi("eigevery", 1),
        "heavyevery": max(1, fi("heavyevery", 4)),   # cadence for the heaviest panels §7-proj/§8 (keeps runs responsive)
        "slqprobes": fi("slqprobes", 4), "energyp": ff("energyp", 99),
        "seed": fi("seed", 0), "start": fi("start", 0), "inputstd": ff("inputstd", 1.0),
        "ssign": g("ssign", "off"), "optimizer": g("optimizer", "gd"), "qapprox": max(1, fi("qapprox", 25)),
        "qmode": max(1, fi("qmode", 1)), "tset": max(1, fi("tset", 3)),
        "cubicapprox": max(1, fi("cubicapprox", 10)),   # §10 cubic-approximation window length
        "dataset": g("dataset", "synthetic"), "arch": g("arch", "mlp"), "loss": g("loss", "mse"),
        "chmul": ff("chmul", 0.25), "nlayer": fi("nlayer", 2), "nhead": fi("nhead", 4),
        "dmodel": fi("dmodel", 64), "seqlen": fi("seqlen", 16), "vocab": fi("vocab", 50257),
        "degree": fi("degree", 3),       # Chebyshev polynomial degree (chebyshev dataset)
        "ksparse": max(1, fi("ksparse", 3)),   # k-sparse parity: #bits in the parity subset S (≤ indim = #input bits)
        "c2a": fi("c2a", 0), "c2b": fi("c2b", 1),   # cifar2: the two CIFAR-10 class indices → scalar labels +1 / −1
        "divreg": ff("divreg", 0.3),     # §15 divergence ε strength: softens the 0/0 spike at D²→0 inflections (0 ⇒ exact /D²)
        "cvar": ff("cvar", 0.0),         # const dataset: variance of Gaussian noise added to the constant target (0 ⇒ exact constant)
        "initscheme": g("initscheme", "default"),
        "surrogate": g("surrogate", "quad"), "mode": g("mode", "run"),
        "s1": g("s1", "1") == "1", "s2": g("s2", "1") == "1", "s3": g("s3", "1") == "1",
        "s4": g("s4", "1") == "1", "s5": g("s5", "1") == "1", "s6": g("s6", "1") == "1",
        "s7": g("s7", "1") == "1", "s8": g("s8", "1") == "1", "s9": g("s9", "1") == "1",
        "s10": g("s10", "1") == "1", "s11": g("s11", "1") == "1", "s12": g("s12", "0") == "1",
        "s13": g("s13", "1") == "1",
        "s14": g("s14", "0") == "1",     # §9c: σ₁ predictions vs full loss-Hessian sharpness
        "s15": g("s15", "1") == "1",     # §9d: predictions with the residual self-computed by the quadratic model
        "s16": g("s16", "1") == "1",     # §9d-c: §9d predictions vs full loss-Hessian sharpness
        "s17": g("s17", "0") == "1",     # §10: CUBIC approximation (Eq-47/51 σ₁ predictions, exact J&Q propagation; OFF by default — heaviest)
        "s18": g("s18", "0") == "1",     # §11: 3D Hessian–NTK grids (multi-sample, small M; OFF by default)
        "s19": g("s19", "0") == "1",     # §12: per-sample Q_i eigenvector cross-similarity (multi-sample, small N; OFF by default)
        "s20": g("s20", "0") == "1",     # §14: Tr(ΔNTK) per-triplet decomposition cubes (multi-sample, small N; OFF by default)
        "s21": g("s21", "0") == "1",     # §13: residual-weighted curvature G1/G2/G3 vs exact ref (own toggle; multi-sample, small N; OFF by default)
        "s22": g("s22", "0") == "1",     # §12b: per-sample projection panels (s19=§12a angles+diag-align; both share the §12 Lanczos)
        "s23": g("s23", "0") == "1",     # §15: 2nd-difference decomposition of ‖J‖²_F & σ₁ (own toggle; multi-sample, small N; OFF by default)
        "s24": g("s24", "0") == "1",     # §16: curvature-aligned per-residual-sign optimizer (standalone; own iteration axis; OFF by default)
        "s24warm": max(0, fi("s24warm", 5)),    # §16: standard-GD warmup steps before the §16 optimization begins
        "s24iter": max(1, fi("s24iter", 250)),  # §16: number of §16 iterations after the warmup
        "s24grid": min(0.5, max(0.01, ff("s24grid", 0.1))),   # §16: (β,s) search grid step (finer ⇒ more descent, costs (1/step)² loss-evals)
        "s24ares": min(0.1, max(0.001, ff("s24ares", 0.01))), # §16: α line-search resolution (finer ⇒ extracts smaller steps near the plateau)
        "s24k": max(1, fi("s24k", 32)),   # §16/§17: # eigvecs per side (top-kdir & bottom-kdir) onto which g₊/g₋ are projected
        "s24base": g("s24base", "0") == "1",  # §16: compute the five dotted baselines (A=random,B=shuffle,C=randfix,D=eigfix,E=gd); ≈6× cost
        "s25": g("s25", "0") == "1",     # §17: PER-SAMPLE function-Hessian variant of §16 (each sample uses its OWN Q_k eigvec; OFF by default)
        "s25base": g("s25base", "0") == "1",  # §17: compute the five dotted baselines (per-sample analog of §16's); ≈6× cost
        "grid3dcap": max(1, fi("grid3dcap", 500)),
        "sec12ncap": max(1, fi("sec12ncap", 500)),
        "cubeevery": max(1, fi("cubeevery", 1)),   # §11-§14 (3D-grid/cube + per-sample) snapshot cadence: emit every k-th eig-tick.
                                                   #   1 = every tick (default); raise it (e.g. 8) to keep long all-sections captures small (the
                                                   #   N³ cubes are the file-size driver) — §15 / §1-§9 / §16-§17 stay full-resolution.
        "gs": g("gson", "1") == "1",
        # surrogate-section panel toggles (loss · resid mean/std · top-n eig · histogram · theory · §4 · §4d)
        "c1": g("c1", "1") == "1", "c2": g("c2", "1") == "1", "c3": g("c3", "1") == "1",
        "c4": g("c4", "1") == "1", "c5": g("c5", "0") == "1", "c6": g("c6", "1") == "1",
        "c7": g("c7", "1") == "1", "c8": g("c8", "1") == "1", "c9": g("c9", "0") == "1",
        "c10": g("c10", "1") == "1", "c11": g("c11", "1") == "1",   # §9d / §9d-c (surrogate)
        "c12": g("c12", "0") == "1",   # §10 cubic approximation (surrogate section; OFF by default — heaviest)
    }


def _sanitize(o):
    """JSON can't carry NaN/Inf that JS JSON.parse will accept → map them to null."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, list):
        return [_sanitize(x) for x in o]
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    return o


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/run":
            self._sse(parse_qs(u.query))
        else:
            self._static(u.path)

    def _static(self, path):
        if path in ("", "/"):
            path = "/index.html"        # unified page: pick "browser" or "GPU backend" via the Compute dropdown
        fp = os.path.normpath(os.path.join(DIR, path.lstrip("/")))
        if not fp.startswith(DIR) or not os.path.isfile(fp):
            self.send_error(404)
            return
        ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
        with open(fp, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")  # always serve the latest page (no stale cache)
        self.end_headers()
        self.wfile.write(data)

    def _sse(self, q):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        P = _parse_params(q)
        dev, tok = acquire_device()                 # auto least-busy GPU; concurrent tabs land on different devices
        _TL.device = dev
        if dev.type == "cuda":
            torch.cuda.set_device(dev)
        P["_token"] = tok
        gen = run_surrogate_compare(P) if P.get("mode") == "surrogate" else run_stream(P)
        try:
            for msg in gen:
                payload = "data: " + json.dumps(_sanitize(msg)) + "\n\n"
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            gen.close()   # client navigated away / pressed Run again → stop compute
        except Exception as e:  # surface compute errors (OOM, etc.) to the page without crashing the server
            oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
            if _dev().type == "cuda":
                try:
                    torch.cuda.empty_cache()       # release the freed blocks back to the GPU / other users
                except Exception:
                    pass
            msg = ("GPU out of memory. Reduce number_of_samples, lower the model size (smaller chmul / "
                   "dmodel / width), or turn off the heavy panels (§4d, §7, §8, §9 / theory). On a shared "
                   "GPU, restart server.py with --device auto to pick the freest one.") if oom else str(e)
            try:
                err = "data: " + json.dumps({"type": "error", "error": msg}) + "\n\n"
                self.wfile.write(err.encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass
        finally:
            if dev.type == "cuda":
                try:
                    with torch.cuda.device(dev):
                        torch.cuda.empty_cache()   # return this run's memory to its GPU between runs
                except Exception:
                    pass
            release_device(dev)


def build_device_pool(devices_arg, device_arg):
    """Devices that runs are assigned to (one GPU each, auto least-busy in acquire_device()).
    devices_arg: 'all' (every visible GPU) | 'cuda:0,cuda:1,...' | None → single device from --device."""
    if devices_arg:
        if devices_arg.strip().lower() == "all":
            if torch.cuda.is_available():
                return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
            return [torch.device("cpu")]
        return [torch.device(s.strip()) for s in devices_arg.split(",") if s.strip()]
    return [pick_device(device_arg)]


def pick_device(pref):
    if pref and pref != "auto":
        return torch.device(pref)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
        except Exception:
            free = 0
        if free > best_free:
            best_free, best = free, i
    return torch.device(f"cuda:{best}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N (single-device default)")
    ap.add_argument("--devices", default=None,
                    help="multi-GPU pool: 'all' (every visible GPU) | comma list 'cuda:0,cuda:1' | "
                         "omitted = just --device. Concurrent runs are auto-assigned least-busy across the pool.")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float64"],
                    help="auto = fp32 on GPU (fast on A6000), fp64 on CPU")
    ap.add_argument("--cifar-dir", default=None,
                    help="directory with CIFAR-10 raw batches (data_batch_1..5); auto-detected if omitted")
    a = ap.parse_args()

    global DEVICE, DEVICE_POOL, DTYPE, CIFAR_DIR
    CIFAR_DIR = a.cifar_dir
    DEVICE_POOL = build_device_pool(a.devices, a.device)
    DEVICE = DEVICE_POOL[0]
    if a.dtype == "auto":
        DTYPE = torch.float32 if _dev().type == "cuda" else torch.float64
    else:
        DTYPE = torch.float32 if a.dtype == "float32" else torch.float64
    torch.set_grad_enabled(False)
    if _dev().type == "cuda":
        torch.cuda.set_device(_dev())
        # the heavy compute is on the GPU; the per-step Lanczos loop is Python-orchestrated, so on a
        # busy login node spawning many CPU threads only thrashes — pin to 1 to keep the GPU fed.
        torch.set_num_threads(1)

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print("EoS / Progressive-Sharpening — GPU backend")
    print(f"  device : {_dev()}   dtype: {str(DTYPE).replace('torch.', '')}")
    if len(DEVICE_POOL) > 1:
        print(f"  pool   : {', '.join(str(d) for d in DEVICE_POOL)}  (concurrent runs auto-assigned least-busy)")
    print(f"  serving: http://{a.host}:{a.port}/   (index.html — set Compute → \"GPU backend\")")
    print(f"  tunnel : ssh -N -L {a.port}:localhost:{a.port} <this-box>"
          f"   then open http://localhost:{a.port}/")
    print("  Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
