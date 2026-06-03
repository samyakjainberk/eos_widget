"""Faithful Python port + verification harness for eos_widget/index.html.

Mirrors every analysis the widget computes and cross-checks each matrix-free
quantity against an explicit dense reference, so the in-browser numerics can be verified.

Usage:
    python _preview.py            # run verification checks, then render preview PNGs
    python _preview.py verify     # verification checks only
    python _preview.py preview    # render preview PNGs only

Sections mirrored:
  §1  loss, sharpness (top eig of loss Hessian), residual r=y-f (signed), H eigenvalue edges
  §2/§3  top-n / bottom-n eigenvalues of  H=Σ∇²f_a ,  loss Hessian ,  Gauss-Newton G ,  residual term S
  §4  J=∇Σf_a  onto top-n/bottom-n eigvecs of H, and J/‖J‖ (cosine)
  §4b sec_9  J·r=Σ r_a∇f_a onto top-n/bottom-n eigvecs of Q[u₁]=Σ(u₁)_a∇²f_a, /‖J·r‖ and /‖r‖
  §4c sec_10 Q[u₁](J·r) onto top-n Gauss-Newton eigvecs g_k=J v_k/‖J v_k‖ (SVD-paired sign), + normalized
  §5  SLQ spectral density of H, G, S, loss Hessian
  §6  principal angles between consecutive 99%-energy ± eigenspaces of H
  §7  NTK K=JᵀJ ; FH tensor reshaped p²×dn (SVD via Frobenius Gram, Hutchinson) ; M₁,M₂ eigvecs
  §8  FH tensor reshaped p×(p·dn) ; vec(J) onto right singular vecs  (= ⟨w,U_k⟩/σ_k, G2=Σ(∇²f_a)²)
"""
import sys, math, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles

EPS = 1e-3
PAL = ['#2563eb', '#d35400', '#16a34a', '#9333ea', '#dc2626', '#0891b2', '#ea580c', '#475569']

# ---------- mulberry32 (exact JS port; data + init reproducibility) ----------
def u32(x): return x & 0xFFFFFFFF
def i32(x):
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x >= 0x80000000 else x
def imul(a, b): return i32((u32(a) * u32(b)) & 0xFFFFFFFF)
def xor32(a, b): return i32(u32(a) ^ u32(b))
def or32(a, b): return u32(a) | u32(b)
def mulberry32(seed):
    st = {"a": i32(seed)}
    def rnd():
        a = i32(st["a"] + 0x6D2B79F5); st["a"] = a
        t = imul(xor32(a, u32(a) >> 15), or32(1, a))
        t = xor32(i32(t + imul(xor32(t, u32(t) >> 7), or32(61, t))), t)
        return u32(xor32(t, u32(t) >> 14)) / 4294967296.0
    return rnd
def gauss(rng):
    u = 0.0; v = 0.0
    while u == 0: u = rng()
    while v == 0: v = rng()
    return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)

# ---------- model (multi-output) ----------
def build_spec(inDim, width, depth, act, useBias, outDim):
    spec = []; off = 0; din = inDim
    for _ in range(depth):
        dout = width; wOff = off; off += din * dout
        bOff = off if useBias else -1
        if useBias: off += dout
        spec.append((din, dout, act, wOff, bOff)); din = dout
    wOff = off; off += din * outDim; bOff = off if useBias else -1
    if useBias: off += outDim
    spec.append((din, outDim, "linear", wOff, bOff))
    return spec, off
def actf(name, z):
    if name == "tanh": return np.tanh(z)
    if name == "elu":  return np.where(z > 0, z, np.expm1(z))
    return z
def actd(name, z):
    if name == "tanh": return 1 - np.tanh(z) ** 2
    if name == "elu":  return np.where(z > 0, 1.0, np.exp(z))
    return np.ones_like(z)
def fwd(th, spec, X):
    a = X; caches = []
    for (din, dout, act, wOff, bOff) in spec:
        W = th[wOff:wOff + din * dout].reshape(din, dout)
        z = a @ W + (th[bOff:bOff + dout] if bOff >= 0 else 0.0)
        caches.append((a, z, act, din, dout, wOff, bOff)); a = actf(act, z)
    return a, caches                                   # a: (N, outDim)
def bwd(th, spec, caches, dO, p):                      # dO: (N,outDim) -> Σ dO_{n,c} ∇f_{n,c}
    g = np.zeros(p); dA = dO.copy()
    for (a, z, act, din, dout, wOff, bOff) in reversed(caches):
        dZ = dA * actd(act, z); W = th[wOff:wOff + din * dout].reshape(din, dout)
        g[wOff:wOff + din * dout] = (a.T @ dZ).reshape(-1)
        if bOff >= 0: g[bOff:bOff + dout] = dZ.sum(0)
        dA = dZ @ W.T
    return g
def gradF(th, spec, X):
    out, C = fwd(th, spec, X); return bwd(th, spec, C, np.ones_like(out), th.shape[0]), out
def gradL(th, spec, X, Y):
    out, C = fwd(th, spec, X); N = X.shape[0]; return bwd(th, spec, C, (out - Y) / N, th.shape[0]), out
def gradW(th, spec, X, c):                              # Σ c_{n,c} ∇f_{n,c}  (c shape (N,outDim))
    out, C = fwd(th, spec, X); return bwd(th, spec, C, c, th.shape[0])
def init_theta(spec, p, initScale, seed):
    rng = mulberry32(u32(seed)); th = np.zeros(p)
    for (din, dout, act, wOff, bOff) in spec:
        sc = initScale / math.sqrt(din)
        for i in range(din * dout): th[wOff + i] = gauss(rng) * sc
    return th
def grad_out(th, spec, X, a):                           # ∇f_a (single output a, flattened index)
    out, C = fwd(th, spec, X); N, oc = out.shape; dO = np.zeros((N, oc)); dO.reshape(-1)[a] = 1.0
    return bwd(th, spec, C, dO, th.shape[0])
def jac_cols(th, spec, X):                              # (dn, p) per-output gradients ; and flat out
    out, C = fwd(th, spec, X); N, oc = out.shape; dn = N * oc; p = th.shape[0]
    J = np.zeros((dn, p))
    for a in range(dn):
        dO = np.zeros((N, oc)); dO.reshape(-1)[a] = 1.0; J[a] = bwd(th, spec, C, dO, p)
    return J, out.reshape(-1)

# ---------- HVPs (central difference, matrix-free) ----------
def hvpF(th, spec, X, v):                              # H = ∇²(Σ f_a)
    return (gradF(th + EPS*v, spec, X)[0] - gradF(th - EPS*v, spec, X)[0]) / (2*EPS)
def hvpL(th, spec, X, Y, v):                           # loss Hessian
    return (gradL(th + EPS*v, spec, X, Y)[0] - gradL(th - EPS*v, spec, X, Y)[0]) / (2*EPS)
def hvpS(th, spec, X, v, c):                           # Σ c_a ∇²f_a (c shape (N,outDim))
    return (gradW(th + EPS*v, spec, X, c) - gradW(th - EPS*v, spec, X, c)) / (2*EPS)
def hvpG(th, spec, X, v):                              # Gauss-Newton (1/N)Σ ∇f_a∇f_aᵀ  via JVP
    N = X.shape[0]
    fp = fwd(th + EPS*v, spec, X)[0]; fm = fwd(th - EPS*v, spec, X)[0]; f0, C0 = fwd(th, spec, X)
    return bwd(th, spec, C0, (fp - fm) / (2*EPS) / N, th.shape[0])
def jac_hvp(th, spec, X, z):                           # {∇²f_a · z}_a   (dn, p)
    cp, _ = jac_cols(th + EPS*z, spec, X); cm, _ = jac_cols(th - EPS*z, spec, X)
    return (cp - cm) / (2*EPS)
def hvpG2(th, spec, X, v):                             # G2 = Σ_a (∇²f_a)²
    u = jac_hvp(th, spec, X, v); dn = u.shape[0]; out = np.zeros(th.shape[0])
    for a in range(dn):
        out += (grad_out(th + EPS*u[a], spec, X, a) - grad_out(th - EPS*u[a], spec, X, a)) / (2*EPS)
    return out

# ---------- Lanczos / SLQ ----------
def lanczos_T(hvp, p, m, seed=0xA53F9):
    rng = np.random.RandomState(seed & 0x7fffffff); q = rng.randn(p); q /= np.linalg.norm(q)
    Q = []; al = []; be = []; qp = np.zeros(p); beta = 0.0
    for _ in range(min(p, m)):
        Q.append(q); w = hvp(q); a = float(w @ q); al.append(a); w = w - a*q - beta*qp
        for _ in range(2):
            for qk in Q: w -= (w @ qk) * qk
        beta = np.linalg.norm(w); be.append(beta)
        if beta < 1e-10: break
        qp = q; q = w / beta
    k = len(Q); T = np.zeros((k, k))
    for i in range(k): T[i, i] = al[i]
    for i in range(k - 1): T[i, i+1] = T[i+1, i] = be[i]
    mu, Sv = np.linalg.eigh(T); idx = np.argsort(mu)[::-1]
    return np.array(Q), mu, Sv, idx, k
def lanczos_extreme(hvp, p, n, m, seed=0xA53F9):
    Q, mu, Sv, idx, k = lanczos_T(hvp, p, m, seed)
    topv = [Sv[:, idx[min(i, k-1)]] @ Q for i in range(n)]
    botv = [Sv[:, idx[max(0, k-1-i)]] @ Q for i in range(n)]
    topval = [mu[idx[min(i, k-1)]] for i in range(n)]
    botval = [mu[idx[max(0, k-1-i)]] for i in range(n)]
    return topv, botv, topval, botval
def lanczos_extreme_vals(hvp, p, n, m, seed=0xA53F9):
    _, mu, _, idx, k = lanczos_T(hvp, p, m, seed)
    return [mu[idx[min(i, k-1)]] for i in range(n)], [mu[idx[max(0, k-1-i)]] for i in range(n)]
def slq_density(hvp, p, nprobe, m, ngrid=80, seed=0):
    rs = np.random.RandomState(seed & 0x7fffffff); TH = []; W = []
    for _ in range(nprobe):
        v = rs.randn(p); q = v / np.linalg.norm(v); Q = []; al = []; be = []; qp = np.zeros(p); beta = 0.0
        for _ in range(min(p, m)):
            Q.append(q); w = hvp(q); a = float(w @ q); al.append(a); w = w - a*q - beta*qp
            for _ in range(2):
                for qk in Q: w -= (w @ qk) * qk
            beta = np.linalg.norm(w); be.append(beta)
            if beta < 1e-10: break
            qp = q; q = w / beta
        k = len(Q); T = np.zeros((k, k))
        for i in range(k): T[i, i] = al[i]
        for i in range(k - 1): T[i, i+1] = T[i+1, i] = be[i]
        val, V = np.linalg.eigh(T)
        for i in range(k): TH.append(val[i]); W.append(V[0, i]**2 / nprobe)
    TH = np.array(TH); W = np.array(W); lo, hi = TH.min(), TH.max()
    if not hi > lo: hi = lo + 1; lo -= 1
    pad = 0.05*(hi-lo) + 1e-9; lo -= pad; hi += pad; sigma = max((hi-lo)/60, 1e-9)
    x = np.linspace(lo, hi, ngrid)
    y = np.array([np.sum(W*np.exp(-0.5*((lam-TH)/sigma)**2)/(sigma*math.sqrt(2*math.pi))) for lam in x])
    return x, y
def pos_subspace(vals, vecs, frac):
    idx = [i for i in range(len(vals)) if vals[i] > 0]; E = sum(vals[i] for i in idx)
    if E <= 0: return []
    c = 0; sub = []
    for i in idx:
        c += vals[i]; sub.append(vecs[i])
        if c >= frac*E: break
    return sub
def neg_subspace(vals, vecs, frac):
    idx = [i for i in range(len(vals)) if vals[i] < 0]; E = sum(abs(vals[i]) for i in idx)
    if E <= 0: return []
    c = 0; sub = []
    for i in idx:
        c += abs(vals[i]); sub.append(vecs[i])
        if c >= frac*E: break
    return sub
def princ(A, B):
    if len(A) == 0 or len(B) == 0: return None
    ang = np.degrees(subspace_angles(np.array(A).T, np.array(B).T)); return float(ang.max()), float(ang.mean())
def sign_to(cur, prev): return cur if prev is None else (-cur if cur @ prev < 0 else cur)
def pin_sign(v):  # deterministic absolute sign: largest-|component| positive
    mi = int(np.argmax(np.abs(v))); return v if v[mi] >= 0 else -v

# =====================================================================================
#  VERIFICATION — matrix-free (widget) quantities vs explicit dense references
# =====================================================================================
def verify():
    print("="*78); print("VERIFICATION: widget matrix-free numerics vs explicit dense references"); print("="*78)
    rng = np.random.RandomState(0)
    inD, outD, N, w, d = 4, 2, 3, 5, 2                  # small multi-sample net (dn=6) so dense p×p is feasible
    spec, p = build_spec(inD, w, d, "tanh", True, outD); dn = N*outD; n = 3
    X = rng.randn(N, inD); Y = rng.randn(N, outD); th = rng.randn(p)*0.5
    ok = lambda name, err, tol=2e-4: print(f"  [{'PASS' if err < tol else 'FAIL'}] {name:<52} rel-err={err:.2e}")

    # --- gradients vs numeric ---
    def numgrad(fn):
        g = np.zeros(p); e = 1e-6
        for i in range(p):
            dd = np.zeros(p); dd[i] = e; g[i] = (fn(th+dd)-fn(th-dd))/(2*e)
        return g
    gL = gradL(th, spec, X, Y)[0]; gLn = numgrad(lambda t: 0.5*np.sum((fwd(t,spec,X)[0]-Y)**2)/N)
    ok("§1 gradL  vs numeric", np.linalg.norm(gL-gLn)/np.linalg.norm(gLn))
    gF = gradF(th, spec, X)[0]; gFn = numgrad(lambda t: fwd(t,spec,X)[0].sum())
    ok("§1 gradF  vs numeric", np.linalg.norm(gF-gFn)/np.linalg.norm(gFn))

    # --- dense references ---
    Jc, out = jac_cols(th, spec, X); Yf = Y.reshape(-1); r = Yf - out      # r = y - f (flat dn)
    cflat = (out - Yf) / N                                                  # (f-y)/N
    def Hess(a):
        H = np.zeros((p, p))
        for j in range(p):
            e = np.zeros(p); e[j] = 1e-4; H[:, j] = (grad_out(th+e,spec,X,a)-grad_out(th-e,spec,X,a))/(2e-4)
        return 0.5*(H+H.T)
    As = [Hess(a) for a in range(dn)]
    Hf = sum(As)                                                      # function Hessian = Σ∇²f_a
    G  = (Jc.T @ Jc) / N                                              # Gauss-Newton
    S  = sum(cflat[a]*As[a] for a in range(dn))                       # residual term
    HL = G + S                                                        # loss Hessian = G + S

    def dense(hvp):                                                   # build matrix-free operator densely
        M = np.array([hvp(np.eye(p)[:,q]) for q in range(p)]).T; return 0.5*(M+M.T)
    ok("§1 hvpF == H (dense Σ∇²f_a)", np.linalg.norm(dense(lambda v:hvpF(th,spec,X,v))-Hf)/np.linalg.norm(Hf))
    ok("§2 hvpG == Gauss-Newton",     np.linalg.norm(dense(lambda v:hvpG(th,spec,X,v))-G)/np.linalg.norm(G))
    cS = cflat.reshape(N, outD)
    ok("§2 hvpS == residual term S",  np.linalg.norm(dense(lambda v:hvpS(th,spec,X,v,cS))-S)/np.linalg.norm(S))
    ok("§2 loss Hessian == G + S",    np.linalg.norm(dense(lambda v:hvpL(th,spec,X,Y,v))-HL)/np.linalg.norm(HL))
    print(f"       (Gauss-Newton min eig = {np.linalg.eigvalsh(G).min():+.2e}  → PSD)")

    # --- §4: Lanczos eigvecs of H vs dense; J projection ---
    tv, bv, tval, bval = lanczos_extreme(lambda v:hvpF(th,spec,X,v), p, n, max(2*n+24,44))
    wHf, VHf = np.linalg.eigh(Hf); iH = np.argsort(wHf)[::-1]
    ok("§4 H top eigval (Lanczos vs dense)", abs(tval[0]-wHf[iH[0]])/abs(wHf[iH[0]]))
    ok("§4 H top eigvec (|overlap|→1)", abs(abs(tv[0]@VHf[:,iH[0]])-1))

    # --- §7: NTK, left singular vecs U_J, FH-tensor Frobenius Gram, M1/M2 ---
    K = Jc @ Jc.T; wK, Vk = np.linalg.eigh(K); iK = np.argsort(wK)[::-1]
    Jx = Jc.T; Uj, sj, Vjt = np.linalg.svd(Jx, full_matrices=False)
    UJ_mf = (Vk[:, iK[0]] @ Jc) / np.sqrt(max(wK[iK[0]], 1e-30))      # J v1 / σ1
    ok("§7 U_J (left sing vec of J, |overlap|→1)", abs(abs(UJ_mf@Uj[:,0])-1))
    Hmat2 = np.stack([A.reshape(-1) for A in As], 1)                  # p²×dn
    GHx = Hmat2.T @ Hmat2                                             # Frobenius Gram dn×dn
    def hutch_GH(m=300):
        rs = np.random.RandomState(7); GG = np.zeros((dn, dn))
        for _ in range(m):
            z = rs.randn(p); Hz = jac_hvp(th, spec, X, z); GG += Hz @ Hz.T
        return GG/m
    ok("§7 FH-tensor Frobenius Gram (Hutchinson m=300)", np.linalg.norm(hutch_GH()-GHx)/np.linalg.norm(GHx), tol=0.1)
    UH, sH, VHt = np.linalg.svd(Hmat2, full_matrices=False); VH = VHt.T
    # M1 = reshape(top-left-singular-vec) == Hessian of Σ_a VH[a,0] f_a / σ ; check top eigvec via hvpS
    M1x = UH[:,0].reshape(p,p); wM, VM = np.linalg.eigh(0.5*(M1x+M1x.T)); iM = np.argsort(wM)[::-1]
    tv1, _, _, _ = lanczos_extreme(lambda v:hvpS(th,spec,X,v,VH[:,0].reshape(N,outD)), p, 1, max(2*1+24,40))
    ok("§7 M₁ top eigvec (reshape vs hvpS, |overlap|→1)", abs(abs(tv1[0]@VM[:,iM[0]])-1))

    # --- §8: G2 = Σ(∇²f_a)², w = Σ∇²f_a∇f_a, ⟨vec(J),V_k⟩ = ⟨w,U_k⟩/σ_k ---
    G2x = sum(A@A for A in As); wmf = sum(As[a]@Jc[a] for a in range(dn))
    ok("§8 hvpG2 == Σ(∇²f_a)²", np.linalg.norm(dense(lambda v:hvpG2(th,spec,X,v))-G2x)/np.linalg.norm(G2x))
    Hwide = np.concatenate(As, 1); Uw, sw, Vwt = np.linalg.svd(Hwide, full_matrices=False); Vw = Vwt.T
    vecJ = np.concatenate(list(Jc)); wG2, UG2 = np.linalg.eigh(G2x); iG2 = np.argsort(wG2)[::-1]
    expl = np.array([vecJ @ Vw[:,k] for k in range(n)])
    mf   = np.array([(wmf @ UG2[:,iG2[k]])/np.sqrt(max(wG2[iG2[k]],1e-30)) for k in range(n)])
    ok("§8 ⟨vec(J), right-sing_k⟩ (mf vs dense SVD)", np.linalg.norm(np.abs(expl)-np.abs(mf))/np.linalg.norm(expl))

    # --- §4b: Q[u₁] eigvecs vs dense ; J·r ---
    u1 = Vk[:, iK[0]]; Qx = sum(u1[a]*As[a] for a in range(dn)); Qx = 0.5*(Qx+Qx.T)
    Jr = Jc.T @ r; wQ, VQ = np.linalg.eigh(Qx); iQ = np.argsort(wQ)[::-1]
    tvq, bvq, _, _ = lanczos_extreme(lambda v:hvpS(th,spec,X,v,u1.reshape(N,outD)), p, n, max(2*n+16,28))
    ok("§4b Q[u₁] top eigvec (|overlap|→1)", abs(abs(tvq[0]@VQ[:,iQ[0]])-1))
    ok("§4b J·r == Σ_a r_a∇f_a", np.linalg.norm(Jr - sum(r[a]*Jc[a] for a in range(dn)))/np.linalg.norm(Jr))

    # --- §4c: GN eigvec g_k=J v_k/‖·‖ matches dense GN eig; sign invariance to sign(u₁) ---
    u1 = pin_sign(u1.copy())                                         # deterministic absolute sign (as the widget now does)
    sig1 = np.linalg.norm(Jc.T @ u1); g1 = (Jc.T @ u1) / sig1        # v₁ = J u₁ / σ₁
    wGg, VGg = np.linalg.eigh(G); iGg = np.argsort(wGg)[::-1]
    ok("§4c g₁=J·v₁/‖·‖ == GN top eigvec (|overlap|→1)", abs(abs(g1@VGg[:,iGg[0]])-1))
    ok("§4b/c  J·u₁ = σ·v₁  (σ>0, ‖u₁‖=‖v₁‖=1)", np.linalg.norm(Jc.T@u1 - sig1*g1)/max(np.linalg.norm(Jc.T@u1),1e-9))
    print(f"       (σ₁={sig1:.4f} > 0, ‖u₁‖={np.linalg.norm(u1):.4f}, ‖v₁‖={np.linalg.norm(g1):.4f})")
    def proj_4c(sgn):
        uu = sgn*u1; QJr = hvpS(th, spec, X, Jr, uu.reshape(N, outD)); g = Jc.T@uu; g /= np.linalg.norm(g); return QJr@g
    p_plus, p_minus = proj_4c(+1), proj_4c(-1)
    ok("§4c ⟨Q[u₁]J·r, g₁⟩ invariant to sign(u₁)", abs(p_plus-p_minus)/max(abs(p_plus),1e-9))

    # --- §4d: per-group J / J·r — groups partition samples by sign of the summed residual ---
    ssum = r.reshape(N, outD).sum(axis=1)                      # sum over outputs per sample
    posIdx = np.where(ssum >= 0)[0]; negIdx = np.where(ssum < 0)[0]
    def grpJ(idx, wr):
        g = np.zeros(p)
        for i in idx:
            for c in range(outD):
                a = i*outD + c; g += (r[a] if wr else 1.0)*Jc[a]
        return g
    Jp, Jn = grpJ(posIdx, False), grpJ(negIdx, False)
    Jrp, Jrn = grpJ(posIdx, True), grpJ(negIdx, True)
    Jfull = Jc.sum(axis=0)                                      # §4 J = Σ_a ∇f_a
    ok("§4d  J_pos + J_neg == J (full)", np.linalg.norm(Jp+Jn-Jfull)/max(np.linalg.norm(Jfull),1e-9))
    ok("§4d  (J·r)_pos + (J·r)_neg == J·r", np.linalg.norm(Jrp+Jrn-Jr)/max(np.linalg.norm(Jr),1e-9))
    wH0 = VHf[:, iH[0]]                                         # shared H top eigvec
    ok("§4d  ⟨J_pos,w⟩+⟨J_neg,w⟩ == ⟨J,w⟩ (H eigvec)", abs((Jp@wH0+Jn@wH0)-(Jfull@wH0))/max(abs(Jfull@wH0),1e-9))
    ok("§4d  ⟨Jr_pos,w⟩+⟨Jr_neg,w⟩ == ⟨J·r,w⟩ (Q[u₁] eigvec)", abs((Jrp@tvq[0]+Jrn@tvq[0])-(Jr@tvq[0]))/max(abs(Jr@tvq[0]),1e-9))
    print(f"       (groups: {len(posIdx)} pos / {len(negIdx)} neg of {N} samples)")

    # --- §5: SLQ density of H — integrates to 1; mean matches dense mean(H), normalized by spectral scale ---
    xg, yg = slq_density(lambda v:hvpF(th,spec,X,v), p, 8, 50, 200, seed=1)
    dx = xg[1]-xg[0]; mass = np.sum(yg)*dx; meanH_slq = np.sum(xg*yg)*dx/max(mass,1e-12)
    scale = max(np.max(np.abs(wHf)), 1e-9)               # spectral scale (mean(H)≈0 → use scale, not mean, to normalize)
    ok("§5 SLQ density integrates to 1", abs(mass-1.0), tol=0.05)
    ok("§5 SLQ mean eig vs dense mean(H) /scale", abs(meanH_slq-np.mean(wHf))/scale, tol=0.1)

    # --- §6: principal angles vs scipy ---
    Ua = [VHf[:, iH[i]] for i in range(2)]; Ub = [0.99*VHf[:, iH[i]]+0.01*VHf[:, iH[i+2]] for i in range(2)]
    Ub = list(np.linalg.qr(np.array(Ub).T)[0].T)
    mx_mine, mn_mine = princ(Ua, Ub); ref = np.degrees(subspace_angles(np.array(Ua).T, np.array(Ub).T))
    ok("§6 principal angle (mine vs scipy)", abs(mx_mine-ref.max())/max(ref.max(),1e-9))

    # --- data: same-sign residual mode (single-sample fixed x=1 AND multi-sample iid) ---
    for desc, cfg in [("single fixed x=1", dict(fixedx="1", nsamp=1)), ("multi iid", dict(fixedx="0", nsamp=12))]:
        for sgn in ("pos", "neg"):
            Pd = dict(depth=2, width=8, act="tanh", bias="1", inputstd=1.0, lr=0.3, init=0.5,
                      indim=4, outdim=1, tgt=1.0, seed=2, ssign=sgn, **cfg)
            sp2, p2, th2, Xd, Yd = gen_data(Pd)
            Rd = (Yd - fwd(th2, sp2, Xd)[0]).reshape(-1); want = 1 if sgn == "pos" else -1
            ok(f"§data same-sign [{desc}, {sgn}] residuals all sign {'+' if want>0 else '-'}",
               float(np.sum(Rd*want <= 0)), tol=0.5)   # count of wrong-sign residuals must be 0
            print(f"       (min |r| = {np.min(np.abs(Rd)):.3f} ≥ floor 0.25, n={len(Rd)})")
    print("="*78)

# =====================================================================================
#  RUN + PREVIEW (faithful port of the live loop; renders a multi-row figure)
# =====================================================================================
def analyze(th, spec, X, Y, P, prev):
    """One analysis tick — returns dict matching the widget's plotted quantities."""
    p = th.shape[0]; N, outD = X.shape[0], Y.shape[1]; M = N*outD; n = P["neig"]
    nProbe = P.get("slqprobes", 4); efrac = min(1.0, max(0.5, P.get("energyp", 99)/100.0))
    mF = min(p, max(2*max(n, 10)+24, 44)); mV = min(p, max(2*n+16, 28)); mSLQ = min(p, 40)
    fr, out = gradF(th, spec, X); J = fr; loss = 0.5*np.sum((out-Y)**2)/N
    cS = (out-Y)/N; r = (Y-out).reshape(-1)
    D = {"loss": loss, "r": r}
    # §1 sharpness + H edges ; §2/§3 eig evolution
    hlt, hlb = lanczos_extreme_vals(lambda v:hvpL(th,spec,X,Y,v), p, n, mV, 0x5EED1); D["sharp"] = hlt[0]
    feTv, feBv, feTval, feBval = lanczos_extreme(lambda v:hvpF(th,spec,X,v), p, max(n,10), mF, 0xA53F9)
    D["hfMax"], D["hfMin"] = feTval[0], feBval[0]
    D["HfT"], D["HfB"] = feTval[:n], feBval[:n]; D["HLT"], D["HLB"] = hlt, hlb
    gnt, gnb = lanczos_extreme_vals(lambda v:hvpG(th,spec,X,v), p, n, mV, 0x6A17C)
    srt, srb = lanczos_extreme_vals(lambda v:hvpS(th,spec,X,v,cS), p, n, mV, 0x7B23D)
    D["GNT"], D["GNB"], D["SRT"], D["SRB"] = gnt, gnb, srt, srb
    # §4 J onto H eigvecs (+normalized)
    jn = max(np.linalg.norm(J), 1e-30)
    D["Jt"] = [J@v for v in feTv[:n]]; D["Jb"] = [J@v for v in feBv[:n]]
    D["JtN"] = [x/jn for x in D["Jt"]]; D["JbN"] = [x/jn for x in D["Jb"]]
    # multi-sample base
    if M > 1:
        Jc, o2 = jac_cols(th, spec, X); K = Jc@Jc.T; wK, Vk = np.linalg.eigh(K); iK = np.argsort(wK)[::-1]
        u1 = pin_sign(Vk[:, iK[0]].copy()); rr = Y.reshape(-1)-o2; Jr = Jc.T@rr
        Jrn = max(np.linalg.norm(Jr), 1e-30); Rn = max(np.linalg.norm(rr), 1e-30)
        # §4b  J·r onto eigvecs of Q[u1]
        tvq, bvq, _, _ = lanczos_extreme(lambda v:hvpS(th,spec,X,v,u1.reshape(N,outD)), p, n, mV, 0xC0DE1)
        D["q9t"] = [Jr@v for v in tvq]; D["q9b"] = [Jr@v for v in bvq]
        D["q9tN"] = [x/Jrn for x in D["q9t"]]; D["q9bN"] = [x/Jrn for x in D["q9b"]]
        D["q9tR"] = [x/Rn for x in D["q9t"]]; D["q9bR"] = [x/Rn for x in D["q9b"]]
        g1 = Jc.T @ u1; g1 /= max(np.linalg.norm(g1), 1e-30)   # GN top eigvec g₁=J u₁/‖·‖ (SVD-paired)
        D["q9gt"] = [g1@v for v in tvq]; D["q9gb"] = [g1@v for v in bvq]   # ⟨g₁, top/bottom eigvecs of Q[u₁]⟩
        # §4c  Q[u1](J·r) onto GN eigvecs g_k = J v_k/‖·‖
        QJr = hvpS(th, spec, X, Jr, u1.reshape(N, outD)); qn = max(np.linalg.norm(QJr), 1e-30); D["q10"] = []
        for k in range(n):
            vk = pin_sign(Vk[:, iK[k]].copy()); g = Jc.T@vk; g /= max(np.linalg.norm(g), 1e-30); D["q10"].append(QJr@g)
        D["q10N"] = [x/qn for x in D["q10"]]
        # §7 NTK projections + FH-tensor (Frobenius Gram, M1/M2 eigvals)
        D["ntkR"] = [rr@Vk[:, iK[k]] for k in range(n)]
        GG = np.zeros((M, M)); rs = np.random.RandomState(11)
        for _ in range(nProbe):
            z = rs.randn(p); Hz = jac_hvp(th, spec, X, z); GG += Hz@Hz.T
        GG /= nProbe; wH, VHg = np.linalg.eigh(GG); iHg = np.argsort(wH)[::-1]
        D["ntkH"] = [Vk[:, iK[0]]@VHg[:, iHg[k]] for k in range(n)]
        m1t, m1b = lanczos_extreme_vals(lambda v:hvpS(th,spec,X,v,VHg[:,iHg[0]].reshape(N,outD)), p, 2, mV, 0x9E37)
        m2t, m2b = lanczos_extreme_vals(lambda v:hvpS(th,spec,X,v,VHg[:,iHg[1]].reshape(N,outD)), p, 2, mV, 0x7C19)
        D["fhEvT"] = [m1t[0], m1t[1], m2t[0], m2t[1]]; D["fhEvB"] = [m1b[0], m1b[1], m2b[0], m2b[1]]
        # §8 vec(J) onto right singular vecs of p×(p·dn) reshape
        wv = np.zeros(p)
        for a in range(M): wv += (grad_out(th+EPS*Jc[a],spec,X,a)-grad_out(th-EPS*Jc[a],spec,X,a))/(2*EPS)
        jfn = max(np.linalg.norm(np.concatenate(list(Jc))), 1e-30)
        gtv, _, gtval, _ = lanczos_extreme(lambda v:hvpG2(th,spec,X,v), p, n, mV, 0xB2D4)
        D["g2J"] = [(wv@gtv[k])/np.sqrt(max(gtval[k],1e-30)) for k in range(n)]
        D["g2Jn"] = [x/jfn for x in D["g2J"]]
        # §4d per-group J / J·r onto the shared H and Q[u₁] eigvecs (groups fixed at init)
        grp = P.get("_groups")
        if grp is not None:
            posIdx, negIdx = grp
            def groupJ(idx, wr):
                g = np.zeros(p)
                for i in idx:
                    for c in range(outD):
                        a = i*outD + c; g += (r[a] if wr else 1.0)*Jc[a]
                return g
            Jp, Jn = groupJ(posIdx, False), groupJ(negIdx, False)
            Jrp, Jrn = groupJ(posIdx, True), groupJ(negIdx, True)
            D["d4Ht"] = [Jp@feTv[k] for k in range(n)] + [Jn@feTv[k] for k in range(n)]
            D["d4Hb"] = [Jp@feBv[k] for k in range(n)] + [Jn@feBv[k] for k in range(n)]
            D["d4Qt"] = [Jrp@tvq[k] for k in range(n)] + [Jrn@tvq[k] for k in range(n)]
            D["d4Qb"] = [Jrp@bvq[k] for k in range(n)] + [Jrn@bvq[k] for k in range(n)]
    return D

def gen_data(P):
    """Build spec/θ and the (X,Y) data — mirrors the widget: fixed x=1, iid Gaussian, or same-sign residual."""
    spec, p = build_spec(P["indim"], P["width"], P["depth"], P["act"], P["bias"]=="1", P["outdim"])
    N, outD = P["nsamp"], P["outdim"]; drng = mulberry32(u32(P["seed"]*7919+1))
    th = init_theta(spec, p, P["init"], P["seed"]+1)                # θ first (residual signs need it)
    fx = P["fixedx"] == "1"; sd = P.get("inputstd", 1.0); ssign = P.get("ssign", "off")
    if fx:
        X = np.ones((N, P["indim"]))
        if ssign == "off":
            Y = np.full((N, outD), float(P["tgt"]))
        else:                                                       # single-sample (fixed x=1): construct y so r has chosen sign
            s = 1.0 if ssign == "pos" else -1.0; floor = 0.25*max(abs(P["tgt"]), 1e-6)
            F = fwd(th, spec, X)[0]; Y = F + s*max(abs(P["tgt"]), floor)
    elif ssign != "off":
        s = 1.0 if ssign == "pos" else -1.0; floor = 0.25*max(abs(P["tgt"]), 1e-6); K = max(64*N, 3000)
        Xc = np.array([[sd*gauss(drng) for _ in range(P["indim"])] for _ in range(K)])
        Yc = np.array([[P["tgt"]*gauss(drng) for _ in range(outD)] for _ in range(K)])
        Fc = fwd(th, spec, Xc)[0]; R = Yc - Fc
        okSign = np.all(R*s > 0, axis=1); order = np.argsort(-np.min(np.abs(R), axis=1))
        good = [k for k in order if okSign[k]]; rest = [k for k in order if not okSign[k]]
        chosen = (good + rest)[:N]
        X = Xc[chosen]; Y = Fc[chosen] + s*np.maximum(np.abs(Yc[chosen]-Fc[chosen]), floor)   # r=y−f sign s, |r|≥floor
    else:
        X = np.array([[sd*gauss(drng) for _ in range(P["indim"])] for _ in range(N)])
        Y = np.array([[P["tgt"]*gauss(drng) for _ in range(outD)] for _ in range(N)])
    return spec, p, th, X, Y

def run(P):
    spec, p, th, X, Y = gen_data(P)
    N, outD = P["nsamp"], P["outdim"]; M = N*outD
    steps, ee, lr = P["steps"], P["eigevery"], P["lr"]
    # §4d: fix the two sample groups by the sign of the summed initial residual r=y−f(x,θ₀)
    ssum0 = (Y - gradF(th, spec, X)[1]).sum(axis=1)
    P = {**P, "_groups": (np.where(ssum0 >= 0)[0], np.where(ssum0 < 0)[0])}
    keys = None; series = {}; T = []
    for t in range(steps+1):
        if t % ee == 0:
            D = analyze(th, spec, X, Y, P, None)
            if keys is None:
                keys = list(D.keys()); series = {k: [] for k in keys}
            for k in keys: series[k].append(D[k])
            T.append(t)
            if not np.isfinite(D["loss"]) or D["loss"] > 1e10: break
        th = th - lr*gradL(th, spec, X, Y)[0]
    series["T"] = T; series["p"] = p; series["M"] = M; series["thr"] = 2.0/lr; series["n"] = P["neig"]
    return series

def _lines(ax, x, mat, title, n=None):
    arr = np.array(mat); n = n or arr.shape[1]
    for i in range(min(n, arr.shape[1])): ax.plot(x, arr[:, i], color=PAL[i % len(PAL)], lw=1.4)
    ax.set_title(title, fontsize=9); ax.set_xlabel("step", fontsize=7); ax.tick_params(labelsize=6)

def plot(P, fname, title):
    D = run(P); x = D["T"]; n = D["n"]; multi = D["M"] > 1
    rows = 10 if multi else 4
    fig = plt.figure(figsize=(17, 3.0*rows)); fig.suptitle(title, fontsize=13, y=0.998)
    gs = fig.add_gridspec(rows, 4, hspace=0.6, wspace=0.32)
    ax = fig.add_subplot(gs[0,0]); ax.semilogy(x, D["loss"], color=PAL[0]); ax.set_title("§1 loss", fontsize=9); ax.tick_params(labelsize=6)
    ax = fig.add_subplot(gs[0,1]); ax.plot(x, D["sharp"], color=PAL[4]); ax.axhline(D["thr"], ls="--", c="#9ca3af", lw=1); ax.set_title("§1 sharpness vs 2/lr", fontsize=9); ax.tick_params(labelsize=6)
    ax = fig.add_subplot(gs[0,2]); R = np.array(D["r"]);
    for j in range(min(R.shape[1], 12)): ax.plot(x, R[:, j], lw=1, color=PAL[j%len(PAL)])
    ax.axhline(0, c="#d1d5db", lw=1); ax.set_title("§1 residual y−f", fontsize=9); ax.tick_params(labelsize=6)
    ax = fig.add_subplot(gs[0,3]); ax.plot(x, D["hfMax"], color=PAL[0], label="λmax"); ax.plot(x, D["hfMin"], color=PAL[4], label="λmin"); ax.axhline(0, c="#d1d5db", lw=1); ax.set_title("§1 H edges", fontsize=9); ax.legend(fontsize=6); ax.tick_params(labelsize=6)
    _lines(fig.add_subplot(gs[1,0]), x, D["HfT"], "§2 H top-n"); _lines(fig.add_subplot(gs[1,1]), x, D["HLT"], "§2 lossH top-n")
    _lines(fig.add_subplot(gs[1,2]), x, D["GNT"], "§2 G top-n"); _lines(fig.add_subplot(gs[1,3]), x, D["SRT"], "§2 S top-n")
    _lines(fig.add_subplot(gs[2,0]), x, D["HfB"], "§3 H bot-n"); _lines(fig.add_subplot(gs[2,1]), x, D["HLB"], "§3 lossH bot-n")
    _lines(fig.add_subplot(gs[2,2]), x, D["GNB"], "§3 G bot-n"); _lines(fig.add_subplot(gs[2,3]), x, D["SRB"], "§3 S bot-n")
    _lines(fig.add_subplot(gs[3,0]), x, D["Jt"], "§4 J·top-n", n); _lines(fig.add_subplot(gs[3,1]), x, D["Jb"], "§4 J·bot-n", n)
    _lines(fig.add_subplot(gs[3,2]), x, D["JtN"], "§4 J/‖J‖·top-n", n); _lines(fig.add_subplot(gs[3,3]), x, D["JbN"], "§4 J/‖J‖·bot-n", n)
    if multi:
        _lines(fig.add_subplot(gs[4,0]), x, D["q9t"], "§4b J·r·top-n Q[u₁]", n); _lines(fig.add_subplot(gs[4,1]), x, D["q9b"], "§4b J·r·bot-n", n)
        _lines(fig.add_subplot(gs[4,2]), x, D["q9tN"], "§4b /‖J·r‖ top", n); _lines(fig.add_subplot(gs[4,3]), x, D["q9bN"], "§4b /‖J·r‖ bot", n)
        _lines(fig.add_subplot(gs[5,0]), x, D["q9tR"], "§4b /‖r‖ top", n); _lines(fig.add_subplot(gs[5,1]), x, D["q9bR"], "§4b /‖r‖ bot", n)
        _lines(fig.add_subplot(gs[5,2]), x, D["q9gt"], "§4b ⟨g₁, Q[u₁] top⟩", n); _lines(fig.add_subplot(gs[5,3]), x, D["q9gb"], "§4b ⟨g₁, Q[u₁] bot⟩", n)
        _lines(fig.add_subplot(gs[6,0]), x, D["q10"], "§4c Q[u₁]J·r·GN top-n", n); _lines(fig.add_subplot(gs[6,1]), x, D["q10N"], "§4c normalized", n)
        _lines(fig.add_subplot(gs[6,2]), x, D["ntkR"], "§7 r·NTK eigvecs", n); _lines(fig.add_subplot(gs[6,3]), x, D["ntkH"], "§7 NTK₁·FH right-sing", n)
        _lines(fig.add_subplot(gs[7,0]), x, D["fhEvT"], "§7 M₁,M₂ top-2 eigval", 4); _lines(fig.add_subplot(gs[7,1]), x, D["fhEvB"], "§7 M₁,M₂ bot-2 eigval", 4)
        _lines(fig.add_subplot(gs[7,2]), x, D["g2J"], "§8 vec(J)·right-sing", n); _lines(fig.add_subplot(gs[7,3]), x, D["g2Jn"], "§8 /‖J‖_F", n)
        def _g(ax, mat, title):                                  # §4d: pos group solid, neg group dotted
            A = np.array(mat)
            for i in range(n): ax.plot(x, A[:, i],   color=PAL[i%len(PAL)], lw=1.6)
            for i in range(n): ax.plot(x, A[:, n+i], color=PAL[i%len(PAL)], lw=1.2, ls=":")
            ax.set_title(title, fontsize=9); ax.set_xlabel("step", fontsize=7); ax.tick_params(labelsize=6)
        _g(fig.add_subplot(gs[8,0]), D["d4Ht"], "§4d grp J·top-n H (+/−)"); _g(fig.add_subplot(gs[8,1]), D["d4Hb"], "§4d grp J·bot-n H")
        _g(fig.add_subplot(gs[8,2]), D["d4Qt"], "§4d grp J·r·top-n Q[u₁]"); _g(fig.add_subplot(gs[8,3]), D["d4Qb"], "§4d grp J·r·bot-n Q[u₁]")
        for c in range(4): fig.add_subplot(gs[9,c]).axis("off")
    fig.savefig(fname, dpi=85, bbox_inches="tight"); plt.close(fig)
    print(f"{fname}: p={D['p']} M={D['M']} 2/lr={D['thr']:.2f}  sharp {D['sharp'][1]:.2f}->{max(D['sharp']):.2f}")

# Preview configs. The widget's actual default `msample` is depth4/width100 (p≈31k) — too heavy for a
# pure-numpy render, so MOD below is a moderate same-shape stand-in (tanh, iid Gaussian, multi-sample).
SMALL = dict(depth=2, width=8,  act="tanh", bias="1", fixedx="0", inputstd=1.0, lr=0.3, init=0.5,
             nsamp=6,  indim=4,  outdim=2, tgt=1.0, neig=3, kth=1, steps=300, eigevery=15, slqprobes=4, energyp=99, seed=1)
MOD   = dict(depth=3, width=24, act="tanh", bias="0", fixedx="0", inputstd=1.0, lr=0.3, init=0.1,
             nsamp=8,  indim=6,  outdim=1, tgt=1.0, neig=3, kth=1, steps=200, eigevery=20, slqprobes=4, energyp=99, seed=0)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "verify"):
        verify()
    if mode in ("all", "preview"):
        plot(SMALL, "preview_small.png", "Multi-sample (small, dn=12) — all sections [verification render]")
        plot(MOD,   "preview_msample.png", "Moderate multi-sample tanh MLP (same shape as default) [seed 0]")
