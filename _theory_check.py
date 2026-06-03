"""Validate the §9 theoretical-prediction formulas (Eq 13 / below-21 / 27 / 29) against the
actual one-step change in the top NTK eigenvalue σ₁.

Correctness criteria (frozen-Q quadratic model ⇒ first order in η):
  * the one-step σ₁ prediction must match the actual one-step σ₁ to O(η²)  → halving η ≈ quarters the error
  * col-3's full FH-tensor sum  Σ_ij γ_i τ_ij ⟨v₁,z_ij⟩² ⟨y_i,u₁⟩  must EXACTLY equal  v₁ᵀ Q[u₁] v₁  (col-2 core)
"""
import numpy as np
import _preview as pv

EPS = pv.EPS

def ntk_eig(th, spec, X):
    Jc, out = pv.jac_cols(th, spec, X)            # (M,p) rows ∇f_a, flat out
    K = Jc @ Jc.T                                  # NTK (M,M)
    w, V = np.linalg.eigh(K); idx = np.argsort(w)[::-1]
    return Jc, out, w[idx], V[:, idx]              # eigvals desc, eigvecs cols (M)

def dense_Hess_a(th, spec, X, a, p):
    """∇²f_a (p,p) by finite-difference of grad_out_a along each coordinate (small p only)."""
    H = np.zeros((p, p))
    for i in range(p):
        e = np.zeros(p); e[i] = 1.0
        H[:, i] = (pv.grad_out(th+EPS*e, spec, X, a) - pv.grad_out(th-EPS*e, spec, X, a))/(2*EPS)
    return 0.5*(H+H.T)

def run():
    # small multi-sample net so dense Hessians are feasible
    P = dict(depth=2, width=6, act="tanh", bias="1", fixedx="0", inputstd=1.0, ssign="off",
             lr=0.3, init=0.5, indim=3, outdim=2, nsamp=5, tgt=1.0, seed=3,
             neig=3, kth=1, steps=0, eigevery=1, slqprobes=4, energyp=99)
    spec, p, th0, X, Y = pv.gen_data(P)
    N, outD = P["nsamp"], P["outdim"]; M = N*outD
    # advance a few steps so we're off the init plateau
    th = th0.copy()
    for _ in range(40): th = th - P["lr"]*pv.gradL(th, spec, X, Y)[0]
    print(f"net: p={p}, M={M}")

    Jc, out, w, V = ntk_eig(th, spec, X)
    sig1 = w[0]; u1 = V[:, 0]                       # σ₁ (top NTK eigval), u₁ (NTK top eigvec, R^M)
    r = (Y.reshape(-1) - out)                       # residual (M,)
    pnorm = np.linalg.norm(r)
    Jr = Jc.T @ r                                   # Jᵀr  (p,)
    sg1 = np.sqrt(max(sig1, 1e-30))                 # singular value √σ₁
    v1 = (Jc.T @ u1); v1 = v1/np.linalg.norm(v1)    # GN top eigvec (p,), J u₁ = σ̃₁ v₁

    # core curvature scalars (matrix-free, as the widget computes them)
    Qu1_v1 = pv.hvpS(th, spec, X, v1, u1.reshape(N, outD))      # Q[u₁] v₁
    c = float(v1 @ Qu1_v1)                          # v₁ᵀ Q[u₁] v₁   (col-2 core)
    Qu1_Jr = pv.hvpS(th, spec, X, Jr, u1.reshape(N, outD))      # Q[u₁](Jᵀr)
    gen_rate = 2*P["lr"]*sg1*float(v1 @ Qu1_Jr)     # GENERAL Δσ₁ (Eq-21, no r̂≈u₁ assumption)
    simp_rate = 2*P["lr"]*pnorm*sig1*c              # SIMPLIFIED Δσ₁ (col-2, assumes r̂≈u₁)
    align = abs(u1 @ (r/pnorm))                     # |⟨r̂, u₁⟩|  (how well Assumption 3 holds)
    print(f"\nresidual–u₁ alignment |⟨r̂,u₁⟩| = {align:.3f}   (col-2/3/4 assume this ≈ 1)")

    # ---- (A) one-step σ₁ prediction vs actual, scaling in η  (GD uses Δθ=(η/N)Jᵀr ⇒ effective η/N) ----
    print("\n(A) one-step σ₁ ratio: prediction vs actual  (general must be O(η²)-accurate ⇒ rel.err ∝ η)")
    print(f"{'η':>9} | {'actual Δσ₁/σ₁':>14} | {'general pred':>12} {'rel.err':>9} | {'simplified':>11} {'rel.err':>9}")
    for eta in (0.05, 0.025, 0.0125, 0.00625, 0.003125):
        eN = eta/N                                   # effective step: Δθ = (η/N) Jᵀr
        thn = th - eta*pv.gradL(th, spec, X, Y)[0]
        _, _, wn, _ = ntk_eig(thn, spec, X)
        act = (wn[0]-sig1)/sig1
        g = 2*eN*sg1*float(v1 @ Qu1_Jr)/sig1         # general ratio  2(η/N)σ̃₁ v₁ᵀQ[u₁](Jᵀr)/σ₁
        s = 2*eN*pnorm*c                              # simplified (r̂≈u₁): 2(η/N) p v₁ᵀQ[u₁]v₁
        eg = abs(g-act)/max(abs(act),1e-12); es = abs(s-act)/max(abs(act),1e-12)
        print(f"{eta:9.5f} | {act:14.3e} | {g:12.3e} {eg:9.2e} | {s:11.3e} {es:9.2e}")

    # ---- (B) col-3 FH-tensor sum == v₁ᵀQ[u₁]v₁ (exact identity, full i,j) ----
    print("\n(B) FH-tensor decomposition identity (col-3 core):")
    As = [dense_Hess_a(th, spec, X, a, p) for a in range(M)]
    # FH tensor T[:,a] = vec(∇²f_a); SVD → y_i (right, R^M), γ_i, X_i = reshape(left)
    T = np.stack([A.reshape(-1) for A in As], axis=1)           # (p², M)
    U, g, Yt = np.linalg.svd(T, full_matrices=False)            # U:(p²,M), g:(M,), Yt:(M,M)
    full = 0.0
    for i in range(M):
        Xi = U[:, i].reshape(p, p); Xi = 0.5*(Xi+Xi.T)
        tau, Z = np.linalg.eigh(Xi)                              # X_i = Σ_j τ_ij z_ij z_ijᵀ
        yu = float(Yt[i] @ u1)
        full += g[i]*sum(tau[j]*(v1 @ Z[:, j])**2 for j in range(p))*yu
    qf_dense = float(v1 @ (sum(u1[a]*As[a] for a in range(M)) @ v1))   # v₁ᵀQ[u₁]v₁ dense
    print(f"   v₁ᵀQ[u₁]v₁  (hvpS) = {c:.6f}")
    print(f"   v₁ᵀQ[u₁]v₁  (dense) = {qf_dense:.6f}   rel.err vs hvpS = {abs(c-qf_dense)/max(abs(qf_dense),1e-12):.2e}")
    print(f"   Σ_ij γτ⟨v₁,z⟩²⟨y,u₁⟩ (full FH) = {full:.6f}   rel.err vs quad form = {abs(full-qf_dense)/max(abs(qf_dense),1e-12):.2e}")

    # ---- (D) col-4 (Eq-29 mode-sum) — full should EQUAL the general; truncated should approximate ----
    fh = []
    for i in range(M):
        Xi = U[:, i].reshape(p, p); Xi = 0.5*(Xi+Xi.T); tau, Z = np.linalg.eigh(Xi); fh.append((g[i], Yt[i], tau, Z))
    def col4_ratio(nv, ni, nj_each, eta):
        eN = eta/N; tot = 0.0
        for vi in range(nv):
            uv = V[:, vi]; sgv = np.sqrt(max(w[vi], 1e-30)); vv = Jc.T@uv; vv = vv/np.linalg.norm(vv)
            rho = float(r @ uv)                                  # residual projection onto NTK mode v
            inner = 0.0
            for i in range(ni):
                gam, yi, tau, Z = fh[i]; yu = float(yi @ u1)
                js = range(p) if nj_each is None else (list(np.argsort(tau)[::-1][:nj_each]) + list(np.argsort(tau)[:nj_each]))
                for j in js:
                    inner += gam*tau[j]*float(v1 @ Z[:, j])*float(vv @ Z[:, j])*yu
            tot += (sgv/sg1)*inner*rho
        return 2*eN*tot
    genr = 2*(0.0125/N)*sg1*float(v1 @ Qu1_Jr)/sig1
    print("\n(D) col-4 (Eq-29) ratio at η=0.0125:")
    print(f"   general (target)                 = {genr:.4e}")
    print(f"   col-4 full (all v, all i,j)      = {col4_ratio(M, M, None, 0.0125):.4e}   rel.err {abs(col4_ratio(M,M,None,0.0125)-genr)/max(abs(genr),1e-12):.2e}")
    print(f"   col-4 trunc (top-3 v, 3 i, ±2 j) = {col4_ratio(3, 3, 2, 0.0125):.4e}   rel.err {abs(col4_ratio(3,3,2,0.0125)-genr)/max(abs(genr),1e-12):.2e}")

    # ---- (C) col-1 single-sample: J-propagation one step vs actual ‖J‖² ----
    print("\n(C) single-sample col-1 (Eq-13 J-propagation), one-step ‖J‖² ratio:")
    Ps = dict(P); Ps.update(nsamp=1, outdim=1, fixedx="1", indim=3)
    sp1, p1, t1, X1, Y1 = pv.gen_data(Ps)
    for _ in range(30): t1 = t1 - Ps["lr"]*pv.gradL(t1, sp1, X1, Y1)[0]
    J1 = pv.gradF(t1, sp1, X1)[0]; out1 = pv.gradF(t1, sp1, X1)[1]
    r1 = float((Y1.reshape(-1)-out1)[0]); nrm0 = float(J1@J1)
    print(f"{'η':>8} | {'actual':>12} {'pred(Eq13)':>12} {'rel.err':>9}")
    for eta in (0.1, 0.05, 0.025):
        tn = t1 - eta*pv.gradL(t1, sp1, X1, Y1)[0]
        Jn = pv.gradF(tn, sp1, X1)[0]; act = (Jn@Jn)/nrm0 - 1
        QJ = pv.hvpF(t1, sp1, X1, J1)                 # Q J  (frozen θ)
        Jp = J1 + eta*r1*QJ; pred = (Jp@Jp)/nrm0 - 1
        print(f"{eta:8.4f} | {act:12.3e} {pred:12.3e} {abs(pred-act)/max(abs(act),1e-12):9.2e}")

def fh_frozen(th, spec, X, nProbe, n7, nM, M, p):
    Jc, _ = pv.jac_cols(th, spec, X)
    GH = np.zeros((M, M)); rs = np.random.RandomState(11)
    for _ in range(nProbe):
        Hz = pv.jac_hvp(th, spec, X, rs.randn(p)); GH += Hz @ Hz.T
    GH /= nProbe; w, V = np.linalg.eigh(GH); idx = np.argsort(w)[::-1]
    F = {"gamma": [], "y": [], "z": [], "tau": []}
    for ii in range(min(n7, M)):
        i = idx[ii]; F["gamma"].append(np.sqrt(max(w[i], 1e-30))); F["y"].append(V[:, i])
        tv, bv, tval, bval = pv.lanczos_extreme(lambda v: pv.hvpS(th, spec, X, v, V[:, i].reshape(X.shape[0], -1)),
                                                p, nM, max(2*nM+16, 28))
        F["z"].append(list(tv)+list(bv)); F["tau"].append(list(tval)+list(bval))
    return F

def trajectory(P, nsteps, qa, fname=None):
    """Run training with the windowed §9 prediction (mirrors the widget); return per-column (pred,actual) series."""
    spec, p, th, X, Y = pv.gen_data(P)
    N, outD = P["nsamp"], P["outdim"]; M = N*outD; lr = P["lr"]; n7 = min(P["neig"], M); nM = 2
    T, cols = [], {k: ([], []) for k in range(4)}                 # cols[c] = (pred[], actual[])
    st = dict(t0=-1)
    aligns = []
    for t in range(nsteps+1):
        etaN = lr/max(N, 1)
        if M > 1:
            Jc, out = pv.jac_cols(th, spec, X); K = Jc@Jc.T; w, V = np.linalg.eigh(K); idx = np.argsort(w)[::-1]
            u1 = pv.pin_sign(V[:, idx[0]]); sig1 = w[idx[0]]; sgT = np.sqrt(max(sig1, 1e-30))
            r = Y.reshape(-1)-out; pn = np.linalg.norm(r); v1 = Jc.T@u1; v1 /= max(np.linalg.norm(v1), 1e-30)
            aligns.append(abs(u1 @ (r/max(pn, 1e-30))))
            if st["t0"] < 0 or (t-st["t0"]) >= qa:
                st.update(t0=t, th0=th.copy(), base=float(sig1), acc2=0.0, p3=1.0, p4=1.0,
                          froz=fh_frozen(th, spec, X, P["slqprobes"], n7, nM, M, p))
            F = st["froz"]; th0 = st["th0"]
            cols[1][0].append(st["base"]+st["acc2"]); cols[1][1].append(sig1)   # col-2 additive
            cols[2][0].append(st["base"]*st["p3"]);   cols[2][1].append(sig1)
            cols[3][0].append(st["base"]*st["p4"]);   cols[3][1].append(sig1)
            def gnv(k): g = Jc.T@pv.pin_sign(V[:, idx[k]]); return g/max(np.linalg.norm(g), 1e-30)
            def fhBil(vk):
                s = 0.0
                for i in range(len(F["gamma"])):
                    yu = float(F["y"][i] @ u1)
                    sj = sum(F["tau"][i][j]*float(v1 @ F["z"][i][j])*float(vk @ F["z"][i][j]) for j in range(len(F["z"][i])))
                    s += F["gamma"][i]*sj*yu
                return s
            # col-2: general Eq-21, additive Δσ with the actual residual (Q[u₁](Jᵀr) at frozen θ₀)
            Jr = Jc.T@r; QJr = pv.hvpS(th0, spec, X, Jr, u1.reshape(N, outD))
            st["acc2"] += 2*etaN*sgT*float(v1 @ QJr)
            # col-3: Eq-27 for a single chosen NTK mode i (residual ∝ u_i)
            mi = min(max(P.get("qmode", 1), 1), min(P["neig"], M)) - 1
            sgi = np.sqrt(max(w[idx[mi]], 1e-30)); phi = float(r @ pv.pin_sign(V[:, idx[mi]]))
            st["p3"] *= (1 + 2*etaN*(sgi/sgT)*phi*fhBil(gnv(mi)))
            # col-4: Eq-29 over the top-|T| NTK modes
            S4 = 0.0
            for vk in range(min(max(P.get("tset", P["neig"]), 1), min(P["neig"], M))):
                sgv = np.sqrt(max(w[idx[vk]], 1e-30)); rho = float(r @ pv.pin_sign(V[:, idx[vk]]))
                S4 += (sgv/sgT)*fhBil(gnv(vk))*rho
            st["p4"] *= (1 + 2*etaN*S4)
        else:
            J = pv.gradF(th, spec, X)[0]; out = pv.gradF(th, spec, X)[1]; sigAct = float(J@J)
            if st["t0"] < 0 or (t-st["t0"]) >= qa:
                st.update(t0=t, th0=th.copy(), Jp=J.copy(), base=sigAct)
            cols[0][0].append(float(st["Jp"]@st["Jp"])); cols[0][1].append(sigAct)
            rsc = float((Y.reshape(-1)-out)[0]); st["Jp"] = st["Jp"] + lr*rsc*pv.hvpF(st["th0"], spec, X, st["Jp"])
        T.append(t)
        th = th - lr*pv.gradL(th, spec, X, Y)[0]
    if fname:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(18, 3.4)); titles = ["col1 single (Eq13)", "col2 multi (∫ Eq21)", "col3 multi (Eq27)", "col4 multi (Eq29)"]
        for c in range(4):
            pr, acc = cols[c]
            if pr:
                ax[c].plot(T[:len(acc)], acc, color="#dc2626", lw=1.6, label="actual σ₁")
                ax[c].plot(T[:len(pr)], pr, color="#2563eb", lw=1.4, label="predicted")
                ax[c].legend(fontsize=7)
            ax[c].set_title(titles[c], fontsize=9); ax[c].set_xlabel("step", fontsize=7); ax[c].tick_params(labelsize=6)
        fig.suptitle(f"§9 theory vs empirical — {fname}  (align⟨r̂,u₁⟩~{np.mean(aligns):.2f})" if aligns else fname)
        fig.savefig(fname, dpi=85, bbox_inches="tight"); plt.close(fig)
    # per-column median relative error (over ticks where prediction has advanced ≥1 step into a window)
    def mederr(c):
        pr, acc = cols[c]
        if not pr: return None
        e = [abs(pr[i]-acc[i])/max(abs(acc[i]), 1e-12) for i in range(len(pr))]
        return float(np.median(e))
    return {c: mederr(c) for c in range(4)}, (float(np.mean(aligns)) if aligns else None)

def scan():
    print("\n================ trajectory scan: per-column median rel-err (pred vs actual σ₁) ================")
    base = dict(neig=3, kth=1, eigevery=1, slqprobes=6, energyp=99, init=0.5, tgt=1.0, bias="1")
    cfgs = [
        ("single fixed, lr0.05, W40", dict(depth=2, width=8, act="tanh", fixedx="1", inputstd=1.0, ssign="off", lr=0.05, indim=3, outdim=1, nsamp=1, seed=2), 200, 40),
        ("multi iid, lr0.05, W40",    dict(depth=2, width=6, act="tanh", fixedx="0", inputstd=1.0, ssign="off", lr=0.05, indim=3, outdim=1, nsamp=4, seed=2), 200, 40),
        ("multi iid, lr0.02, W25",    dict(depth=2, width=6, act="tanh", fixedx="0", inputstd=1.0, ssign="off", lr=0.02, indim=3, outdim=1, nsamp=4, seed=2), 200, 25),
        ("multi +resid, lr0.02, W25", dict(depth=2, width=6, act="tanh", fixedx="0", inputstd=1.0, ssign="pos", lr=0.02, indim=3, outdim=1, nsamp=4, seed=2), 200, 25),
        ("multi 1sample-ish, lr0.02", dict(depth=2, width=6, act="tanh", fixedx="0", inputstd=1.0, ssign="pos", lr=0.02, indim=3, outdim=2, nsamp=2, seed=5), 200, 25),
    ]
    for name, extra, ns, qa in cfgs:
        P = dict(base); P.update(extra); P["steps"] = ns
        errs, align = trajectory(P, ns, qa)
        s = "  ".join(f"col{c+1}={errs[c]:.2e}" if errs[c] is not None else f"col{c+1}=  -  " for c in range(4))
        print(f"  {name:30s} align={align if align is None else round(align,2)} | {s}")

if __name__ == "__main__":
    run()
    scan()
