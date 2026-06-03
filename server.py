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
import json
import math
import mimetypes
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch

DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cpu")   # set in main()
DTYPE = torch.float64          # set in main()
EPS = 1e-3                     # finite-difference step (matches index.html / _preview.py)
PMAX = 1_000_000               # hard cap on parameter count p (matrix-free, so memory is O(p))
RUN_TOKEN = 0                  # monotonic; each new /run claims it so only the latest stream computes
                              # (the server is multi-threaded — without this, stale/reconnected streams
                              #  pile up and thrash the single GPU). Older runs see a newer token and stop.


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
    return z


def actd(name, z):
    if name == "tanh":
        t = torch.tanh(z)
        return 1 - t * t
    if name == "elu":
        return torch.where(z > 0, torch.ones_like(z), torch.exp(z))
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


def gradF(th, spec, X):
    out, C = fwd(th, spec, X)
    return bwd(th, spec, C, torch.ones_like(out)), out


def gradL(th, spec, X, Y):
    out, C = fwd(th, spec, X)
    N = X.shape[0]
    return bwd(th, spec, C, (out - Y) / N), out


def gradW(th, spec, X, c):
    out, C = fwd(th, spec, X)
    return bwd(th, spec, C, c)


def init_theta(spec, p, initScale, seed):
    rng = mulberry32(u32(seed))
    th = [0.0] * p
    for (din, dout, act, wOff, bOff) in spec:
        sc = initScale / math.sqrt(din)
        for i in range(din * dout):
            th[wOff + i] = gauss(rng) * sc
    return torch.tensor(th, dtype=DTYPE, device=DEVICE)


def hvpF(th, spec, X, v):
    return (gradF(th + EPS * v, spec, X)[0] - gradF(th - EPS * v, spec, X)[0]) / (2 * EPS)


def hvpL(th, spec, X, Y, v):
    return (gradL(th + EPS * v, spec, X, Y)[0] - gradL(th - EPS * v, spec, X, Y)[0]) / (2 * EPS)


def hvpS(th, spec, X, v, c):
    return (gradW(th + EPS * v, spec, X, c) - gradW(th - EPS * v, spec, X, c)) / (2 * EPS)


def hvpG(th, spec, X, v):
    N = X.shape[0]
    fp = fwd(th + EPS * v, spec, X)[0]
    fm = fwd(th - EPS * v, spec, X)[0]
    f0, C0 = fwd(th, spec, X)
    return bwd(th, spec, C0, (fp - fm) / (2 * EPS) / N)


def grad_out(th, spec, X, a):
    """∇f_a — gradient of a single (flattened) output index a."""
    out, C = fwd(th, spec, X)
    N, oc = out.shape
    dO = torch.zeros(N, oc, dtype=th.dtype, device=th.device)
    dO.reshape(-1)[a] = 1.0
    return bwd(th, spec, C, dO)


def jac_cols(th, spec, X):
    """Per-output gradients: J of shape (M, p) with row a = ∇f_a; plus flat out (M,).
    One batched backward over all M outputs (the M one-hot seeds = identity), so the whole
    Jacobian is computed in parallel across samples/outputs instead of an M-long Python loop."""
    out, C = fwd(th, spec, X)
    N, oc = out.shape
    M = N * oc
    dO = torch.eye(M, dtype=th.dtype, device=th.device).reshape(M, N, oc)
    J = bwd_batch(th, spec, C, dO)
    return J, out.reshape(-1)


def jac_hvp(th, spec, X, z):
    """{∇²f_a · z}_a — central difference of jac_cols. Shape (M, p)."""
    cp, _ = jac_cols(th + EPS * z, spec, X)
    cm, _ = jac_cols(th - EPS * z, spec, X)
    return (cp - cm) / (2 * EPS)


def hvpG2(th, spec, X, v):
    """G2 = Σ_a (∇²f_a)² applied to v."""
    u = jac_hvp(th, spec, X, v)
    out = torch.zeros(th.shape[0], dtype=th.dtype, device=th.device)
    for a in range(u.shape[0]):
        out += (grad_out(th + EPS * u[a], spec, X, a)
                - grad_out(th - EPS * u[a], spec, X, a)) / (2 * EPS)
    return out


def pin_sign(v):
    """Deterministic absolute sign for an eigenvector: largest-|component| positive."""
    mi = int(torch.argmax(torch.abs(v)))
    return v if float(v[mi]) >= 0 else -v


def sym_eig_desc(A):
    """Eigen-decomposition of a small symmetric matrix, descending. Returns (vals, vecs cols)."""
    w, V = torch.linalg.eigh(A)
    idx = torch.argsort(w, descending=True)
    return w[idx], V[:, idx]


def fh_frozen(th, spec, X, nProbe, n7, nM, mV, M):
    """§9 frozen function-Hessian-tensor SVD at θ: returns (gamma[i], y[i] R^M, z[i][j] R^p, tau[i][j])."""
    Jc, _ = jac_cols(th, spec, X)
    GH = torch.zeros(M, M, dtype=DTYPE, device=DEVICE)
    for pr in range(nProbe):
        Hz = jac_hvp(th, spec, X, _randn_vec(Jc.shape[1], (0xC0FFEE + pr * 40503) & 0xFFFFFFFF))
        GH += Hz @ Hz.t()
    GH /= nProbe
    Hv, Vh = sym_eig_desc(GH)                       # right singular vecs y_i = Vh[:,i]
    nn = min(n7, M)
    out = {"gamma": [], "y": [], "z": [], "tau": []}
    for i in range(nn):
        out["gamma"].append(math.sqrt(max(float(Hv[i]), 1e-30)))
        out["y"].append(Vh[:, i])
        tv, bv, tval, bval = lanczos_extreme(
            lambda v: hvpS(th, spec, X, v, Vh[:, i].reshape(X.shape[0], -1)), Jc.shape[1], nM, mV,
            (0x9E37 + i * 0x1000) & 0xFFFFFFFF)
        out["z"].append(list(tv) + list(bv))
        out["tau"].append(list(tval) + list(bval))
    return out


# ===================== Lanczos / SLQ =====================
def _randn_vec(p, seed):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(int(seed) & 0x7FFFFFFF)
    return torch.randn(p, dtype=DTYPE, device=DEVICE, generator=g)


def _lanczos_core(hvp, p, m, seed):
    q = _randn_vec(p, seed)
    q = q / q.norm()
    Q = []
    al = []
    be = []
    qp = torch.zeros(p, dtype=DTYPE, device=DEVICE)
    beta = torch.zeros((), dtype=DTYPE, device=DEVICE)
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
    mu, Sv = torch.linalg.eigh(T)            # ascending, on CPU float64
    idx = torch.argsort(mu, descending=True)
    return mu, Sv, idx


def lanczos_extreme(hvp, p, n, m, seed):
    Q, T, k = _lanczos_core(hvp, p, m, seed)
    mu, Sv, idx = _eig_sorted(T)
    Qmat = torch.stack(Q)                     # (k, p) on DEVICE

    def ritz(col):
        c = Sv[:, col].to(device=DEVICE, dtype=DTYPE)
        return c @ Qmat                       # (p,)

    topVecs = [ritz(int(idx[min(i, k - 1)])) for i in range(n)]
    botVecs = [ritz(int(idx[max(0, k - 1 - i)])) for i in range(n)]
    topVals = [float(mu[int(idx[min(i, k - 1)])]) for i in range(n)]
    botVals = [float(mu[int(idx[max(0, k - 1 - i)])]) for i in range(n)]
    return topVecs, botVecs, topVals, botVals


def lanczos_extreme_vals(hvp, p, n, m, seed):
    _, T, k = _lanczos_core(hvp, p, m, seed)
    mu, _, idx = _eig_sorted(T)
    top = [float(mu[int(idx[min(i, k - 1)])]) for i in range(n)]
    bot = [float(mu[int(idx[max(0, k - 1 - i)])]) for i in range(n)]
    return top, bot


def slq_density(hvp, p, nprobe, m, ngrid, seed):
    import numpy as np
    TH = []
    W = []
    for pr in range(nprobe):
        _, T, k = _lanczos_core(hvp, p, min(p, m), (seed + pr * 1315423) & 0xFFFFFFFF)
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
    vals = torch.linalg.eigvalsh(G)           # ascending
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


# ===================== streaming run =====================
def run_stream(P):
    """Yield SSE message dicts: one 'meta', then ('step' [+ 'slq']) per eig-tick, then 'done'."""
    spec, p = build_spec(P["indim"], P["width"], P["depth"], P["act"],
                         P["bias"] == "1", P["outdim"])
    half = max(1, p // 2)
    n = min(max(1, P["neig"]), half)
    kth = min(max(1, P["kth"]), half)
    s1, s2, s3, s4, s5, s6, gs = (P["s1"], P["s2"], P["s3"], P["s4"],
                                  P["s5"], P["s6"], P["gs"])
    s7, s8, s9, s10, s11, s12 = P["s7"], P["s8"], P["s9"], P["s10"], P["s11"], P["s12"]
    nSub = min(max(n, kth, (10 if s6 else 1)), half)
    N, inD, outD = P["nsamp"], P["indim"], P["outdim"]
    M = N * outD
    multi = M > 1
    n7 = min(n, max(M, 1))
    n8 = min(n, max(1, p // 2))
    nResid = min(M, 12)
    lr = P["lr"]
    thr = 2.0 / lr

    if p > PMAX:
        yield {"type": "meta", "error": f"p={p} parameters exceeds the cap ({PMAX}). Reduce width/depth."}
        return
    yield {"type": "meta", "p": p, "n": n, "kth": kth, "nResid": nResid, "thr": thr,
           "n7": n7, "n8": n8, "multi": multi,
           "device": f"{DEVICE} · {str(DTYPE).replace('torch.', '')}"}

    # ---- data + init: exact mulberry32 → identical GD trajectory to the browser ----
    th = init_theta(spec, p, P["init"], P["seed"] + 1)         # θ first (residual signs need it)
    drng = mulberry32(u32(P["seed"] * 7919 + 1))
    ssign = P.get("ssign", "off")
    tgt = float(P["tgt"])
    inStd = float(P["inputstd"])
    fixedx = P["fixedx"] == "1"
    if fixedx:
        X = torch.ones(N, inD, dtype=DTYPE, device=DEVICE)
        if ssign == "off":
            Y = torch.full((N, outD), tgt, dtype=DTYPE, device=DEVICE)
        else:                                                  # construct Y so r=y−f has a fixed sign, |r|≥floor
            s = 1.0 if ssign == "pos" else -1.0
            floor = 0.25 * max(abs(tgt), 1e-6)
            F = fwd(th, spec, X)[0]
            Y = F + s * torch.clamp(torch.full_like(F, abs(tgt)), min=floor)
    elif ssign == "off":
        Xl = [[inStd * gauss(drng) for _ in range(inD)] for _ in range(N)]
        Yl = [[tgt * gauss(drng) for _ in range(outD)] for _ in range(N)]
        X = torch.tensor(Xl, dtype=DTYPE, device=DEVICE)
        Y = torch.tensor(Yl, dtype=DTYPE, device=DEVICE)
    else:
        # sample a pool, keep N whose initial residual r=y−f(x,θ₀) is sign-definite with largest |r|
        s = 1.0 if ssign == "pos" else -1.0
        floor = 0.25 * max(abs(tgt), 1e-6)
        K = max(64 * N, 3000)
        Xc = torch.tensor([[inStd * gauss(drng) for _ in range(inD)] for _ in range(K)], dtype=DTYPE, device=DEVICE)
        Yc = torch.tensor([[tgt * gauss(drng) for _ in range(outD)] for _ in range(K)], dtype=DTYPE, device=DEVICE)
        Fc = fwd(th, spec, Xc)[0]
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
        X = torch.empty(N, inD, dtype=DTYPE, device=DEVICE)
        Y = torch.empty(N, outD, dtype=DTYPE, device=DEVICE)
        for i, k in enumerate(chosen):
            X[i] = Xc[k]
            f = Fc[k]
            Y[i] = f + s * torch.clamp((Yc[k] - f).abs(), min=floor)

    # §4d: fix the two sample groups by the sign of the summed initial residual r=y−f(x,θ₀)
    ssum0 = (Y - fwd(th, spec, X)[0]).sum(dim=1)
    pos_rows = torch.tensor([i * outD + c for i in range(N) if float(ssum0[i]) >= 0 for c in range(outD)],
                            dtype=torch.long, device=DEVICE)
    neg_rows = torch.tensor([i * outD + c for i in range(N) if float(ssum0[i]) < 0 for c in range(outD)],
                            dtype=torch.long, device=DEVICE)

    mF = min(p, max(2 * nSub + 24, 44))
    mV = min(p, max(2 * n + 16, 28))
    mSLQ = min(p, 40)
    steps = P["steps"]
    ee = P["eigevery"]
    nProbe = max(1, P["slqprobes"])
    efrac = min(1.0, max(0.5, P["energyp"] / 100.0))
    slqStride = max(1, math.ceil((steps // ee + 1) / 50))
    start = max(0, min(int(P.get("start", 0)), steps))  # resume: fast-forward GD to here, then stream

    global RUN_TOKEN
    RUN_TOKEN += 1
    mytok = RUN_TOKEN                                    # this run is current; a newer /run will bump RUN_TOKEN

    t0 = time.time()
    eigTick = 0
    done = 0
    prevTop = [None] * nSub
    prevBot = [None] * nSub
    prevPos = None
    prevNeg = None
    prevNtk = [None] * n7
    prevUJ = [None] * n7
    prevVH = [None] * n7
    prevM1 = [None] * 4
    prevM2 = [None] * 4
    prevG2 = [None] * n8
    prevQ9t = [None] * n
    prevQ9b = [None] * n
    qapprox = max(1, P["qapprox"]); qmode = P["qmode"]; tset = P["tset"]   # §9 window + Eq-27 mode i + Eq-29 |T|
    thT0 = -1; thTh0 = None; thBase = 0.0; thJp = None; thJ = None; thFroz = None
    thAcc1 = 0.0; thAcc2 = 0.0; thProd3 = 1.0; thProd4 = 1.0

    for t in range(steps + 1):
        if mytok != RUN_TOKEN:
            return            # a newer /run started → drop this stale stream so it stops using the GPU
        if t == start:
            t0 = time.time()  # reset rate clock once streaming begins (exclude fast-forward)
        if t >= start and t % ee == 0:
            J, out = gradF(th, spec, X)
            loss = float(0.5 * ((out - Y) ** 2).sum() / N)
            if not math.isfinite(loss) or loss > 1e10:
                yield {"type": "error",
                       "error": f"diverged at step {t} (loss={loss:.2e}). Lower lr or init_scale."}
                return
            r = (Y - out).reshape(-1)
            cS = (out - Y) / N

            needVecs = s4 or s6
            needVals = s1 or s2 or s3 or needVecs
            feTop, feBot, feTV, feBV = [], [], None, None
            if needVecs:
                feTV, feBV, feTop, feBot = lanczos_extreme(
                    lambda v: hvpF(th, spec, X, v), p, nSub, mF, 0xA53F9)
            elif needVals:
                feTop, feBot = lanczos_extreme_vals(
                    lambda v: hvpF(th, spec, X, v), p, n, mV, 0xA53F9)

            hlt = hlb = None
            if s1 or s2 or s3:
                hlt, hlb = lanczos_extreme_vals(
                    lambda v: hvpL(th, spec, X, Y, v), p, n, mV, 0x5EED1)
            sharp = hlt[0] if hlt else None

            gnt = gnb = srt = srb = None
            if (s2 or s3) and gs:
                gnt, gnb = lanczos_extreme_vals(
                    lambda v: hvpG(th, spec, X, v), p, n, mV, 0x6A17C)
                srt, srb = lanczos_extreme_vals(
                    lambda v: hvpS(th, spec, X, v, cS), p, n, mV, 0x7B23D)

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
            if multi and (s7 or s8 or s9 or s10 or s11 or s12):
                Jc, out_flat = jac_cols(th, spec, X)
                rr = Y.reshape(-1) - out_flat

            # §7 NTK + function-Hessian tensor SVD
            ntkR = ntkH = fhEvT = fhEvB = None
            jhe1 = jhe2 = jhe1b = jhe2b = jh2e1 = jh2e2 = jh2e1b = jh2e2b = None
            if s7 and multi:
                Kv, Vk = sym_eig_desc(Jc @ Jc.t())                 # NTK eigen
                for k in range(n7):
                    Vk[:, k] = sign_to(Vk[:, k], prevNtk[k]); prevNtk[k] = Vk[:, k]
                UJ = []                                            # left singular vecs of J (param space)
                for k in range(n7):
                    u = (Jc.t() @ Vk[:, k]) / math.sqrt(max(float(Kv[k]), 1e-30))
                    u = sign_to(u, prevUJ[k]); prevUJ[k] = u; UJ.append(u)
                GH = torch.zeros(M, M, dtype=DTYPE, device=DEVICE)  # Hessian Frobenius Gram (Hutchinson)
                for pr in range(nProbe):
                    Hz = jac_hvp(th, spec, X, _randn_vec(p, (0xC0FFEE + pr * 40503) & 0xFFFFFFFF))
                    GH += Hz @ Hz.t()
                GH /= nProbe
                Hv, Vh = sym_eig_desc(GH)                          # right singular vecs of FH tensor
                for k in range(n7):
                    Vh[:, k] = sign_to(Vh[:, k], prevVH[k]); prevVH[k] = Vh[:, k]
                nM = min(2, max(1, p // 2))
                fe1t, fe1b, fe1tv, fe1bv = lanczos_extreme(
                    lambda v: hvpS(th, spec, X, v, Vh[:, 0].reshape(N, outD)), p, nM, mV, 0x9E37)
                gh1 = Vh[:, 1] if Vh.shape[1] > 1 else Vh[:, 0]
                fe2t, fe2b, fe2tv, fe2bv = lanczos_extreme(
                    lambda v: hvpS(th, spec, X, v, gh1.reshape(N, outD)), p, nM, mV, 0x7C19)
                def _l(a, i):
                    return a[i] if i < len(a) else a[0]
                m1 = [fe1t[0], _l(fe1t, 1), fe1b[0], _l(fe1b, 1)]
                m2 = [fe2t[0], _l(fe2t, 1), fe2b[0], _l(fe2b, 1)]
                for j in range(4):
                    m1[j] = sign_to(m1[j], prevM1[j]); prevM1[j] = m1[j]
                    m2[j] = sign_to(m2[j], prevM2[j]); prevM2[j] = m2[j]
                fhEvT = [fe1tv[0], _l(fe1tv, 1), fe2tv[0], _l(fe2tv, 1)]
                fhEvB = [fe1bv[0], _l(fe1bv, 1), fe2bv[0], _l(fe2bv, 1)]
                ntkR = [float(rr @ Vk[:, k]) for k in range(n7)]
                ntkH = [float(Vk[:, 0] @ Vh[:, k]) for k in range(n7)]
                jhe1 = [float(UJ[k] @ m1[0]) for k in range(n7)]
                jhe2 = [float(UJ[k] @ m1[1]) for k in range(n7)]
                jhe1b = [float(UJ[k] @ m1[2]) for k in range(n7)]
                jhe2b = [float(UJ[k] @ m1[3]) for k in range(n7)]
                jh2e1 = [float(UJ[k] @ m2[0]) for k in range(n7)]
                jh2e2 = [float(UJ[k] @ m2[1]) for k in range(n7)]
                jh2e1b = [float(UJ[k] @ m2[2]) for k in range(n7)]
                jh2e2b = [float(UJ[k] @ m2[3]) for k in range(n7)]

            # §8 vec(J) onto right singular vecs of the p×(p·dₙ) FH reshape
            g2J = g2Jn = None
            if s8 and multi:
                jfn = max(float(Jc.norm()), 1e-30)                 # ‖J‖_F
                wv = torch.zeros(p, dtype=DTYPE, device=DEVICE)
                for a in range(M):
                    wv += (grad_out(th + EPS * Jc[a], spec, X, a)
                           - grad_out(th - EPS * Jc[a], spec, X, a)) / (2 * EPS)
                gt, _, gtv, _ = lanczos_extreme(lambda v: hvpG2(th, spec, X, v), p, n8, mV, 0xB2D4)
                g2J = []
                for k in range(n8):
                    uk = sign_to(gt[k], prevG2[k]); prevG2[k] = uk
                    g2J.append(float(wv @ uk) / math.sqrt(max(gtv[k], 1e-30)))
                g2Jn = [x / jfn for x in g2J]

            # ---- §4b/§4c/§4d base: NTK top eigvec u₁ (deterministic sign), J·r ----
            u1s = Jrs = bEk_vecs = bEk_vals = None
            Jrns = Rn = 1.0
            if multi and (s9 or s10 or s11 or s12):
                Kb, Vb = sym_eig_desc(Jc @ Jc.t())
                for k in range(n):
                    Vb[:, k] = pin_sign(Vb[:, k])
                bEk_vecs = Vb; bEk_vals = Kb
                u1s = Vb[:, 0]
                Rn = max(float(rr.norm()), 1e-30)
                Jrs = Jc.t() @ rr
                Jrns = max(float(Jrs.norm()), 1e-30)

            # Q[u₁] eigvecs (shared by §4b and §4d), sign-tracked once
            qeT = qeB = None
            if multi and (s9 or s11):
                qeT, qeB, _, _ = lanczos_extreme(
                    lambda v: hvpS(th, spec, X, v, u1s.reshape(N, outD)), p, n, mV, 0xC0DE1)
                for k in range(n):
                    qeT[k] = sign_to(qeT[k], prevQ9t[k]); prevQ9t[k] = qeT[k]
                    qeB[k] = sign_to(qeB[k], prevQ9b[k]); prevQ9b[k] = qeB[k]

            # §4b J·r onto eigvecs of Q[u₁]; alignment with GN top eigvec g₁=J u₁/‖J u₁‖
            q9t = q9b = q9tN = q9bN = q9tR = q9bR = q9gt = q9gb = None
            if s9 and multi:
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
            if s10 and multi:
                QJr = hvpS(th, spec, X, Jrs, u1s.reshape(N, outD))
                qn = max(float(QJr.norm()), 1e-30)
                q10 = []
                for k in range(n):
                    g = Jc.t() @ bEk_vecs[:, k]
                    g = g / max(float(g.norm()), 1e-30)
                    q10.append(float(QJr @ g))
                q10N = [x / qn for x in q10]

            # §4d per-group J / J·r onto the shared H and Q[u₁] eigvecs
            d4Ht = d4Hb = d4Qt = d4Qb = None
            if s11 and multi and feTV is not None and qeT is not None:
                def _grp(rows, weighted):
                    if rows.numel() == 0:
                        return torch.zeros(p, dtype=DTYPE, device=DEVICE)
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
            thP = thA = None
            if s12:
                etaN = lr / max(N, 1); reps_ = max(1, ee)
                if thT0 < 0 or (t - thT0) >= qapprox:               # window start: freeze θ₀, J₀, FH; reset accumulators
                    thT0 = t; thTh0 = th.clone(); thAcc1 = 0.0; thAcc2 = 0.0; thProd3 = 1.0; thProd4 = 1.0
                    if multi:
                        thJ, _ = jac_cols(th, spec, X)                  # predicted Jacobian J₀ (M, p)
                        thFroz = fh_frozen(th, spec, X, nProbe, min(n, M), 2, mV, M)
                        thBase = float(torch.linalg.eigvalsh(thJ @ thJ.t())[-1])
                    else:
                        Jc0, _ = gradF(th, spec, X); thJp = Jc0.clone(); thBase = float(Jc0 @ Jc0)
                thP = [None, None, None, None]; thA = [None, None, None, None]
                if multi and thJ is not None and thFroz is not None and bEk_vals is not None:
                    Jt = thJ; F = thFroz
                    Kw, Vw = sym_eig_desc(Jt @ Jt.t())                  # propagated NTK eigen
                    NV0 = min(n, M)
                    for k in range(NV0):
                        Vw[:, k] = pin_sign(Vw[:, k])
                    sig1 = max(float(Kw[0]), 1e-30); sgT = math.sqrt(sig1); u1 = Vw[:, 0]
                    v1 = Jt.t() @ u1; v1 = v1 / max(float(v1.norm()), 1e-30)
                    sigAct = float(bEk_vals[0]); thA[1] = thA[2] = thA[3] = sigAct
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
                    thP[2] = clmp(thBase*thProd3); thP[3] = clmp(thBase*thProd4)
                    mi = min(max(qmode, 1), NV0) - 1                    # col-3 (Eq-27): single NTK mode i
                    sgi = math.sqrt(max(float(Kw[mi]), 1e-30)); phi = float(rr @ Vw[:, mi])
                    thProd3 *= (1 + 2 * etaN * (sgi/sgT) * phi * fhBil(gnv(mi))) ** reps_
                    S4 = 0.0; NV = min(max(tset, 1), NV0)              # col-4 (Eq-29): sum over top-|T| modes
                    for vk in range(NV):
                        sgv = math.sqrt(max(float(Kw[vk]), 1e-30)); rho = float(rr @ Vw[:, vk])
                        S4 += (sgv/sgT) * fhBil(gnv(vk)) * rho
                    thProd4 *= (1 + 2 * etaN * S4) ** reps_
                    for _ in range(reps_):                             # advance Eq-15: J += (η/N) Q[Jᵀr], Q frozen at θ₀
                        QW = jac_hvp(thTh0, spec, X, thJ.t() @ rr)      # {∇²f_a·(Jᵀr)} at θ₀
                        qu = (u1.unsqueeze(1) * QW).sum(0)              # Q[u₁](Jᵀr) = Σ_a u₁_a QW[a]
                        thAcc2 += 2 * etaN * sgT * float(v1 @ qu)       # col-2 Δσ (first-order, flips with residual sign)
                        thJ = thJ + etaN * QW
                elif not multi and thJp is not None:
                    # col-1 (Eq-13): additive first-order Δσ = 2η r (JᵀQ J) on propagated J — flips with residual sign
                    Jcur, ocur = gradF(th, spec, X); sigAct = float(Jcur @ Jcur); thA[0] = sigAct
                    thP[0] = min(max(thBase + thAcc1, 0.0), 10*max(sigAct, 1e-30)); rsc = float((Y.reshape(-1) - ocur)[0])
                    for _ in range(reps_):
                        QJ = hvpF(thTh0, spec, X, thJp); thAcc1 += 2 * lr * rsc * float(thJp @ QJ); thJp = thJp + lr * rsc * QJ

            yield {
                "type": "step", "t": t, "steps": steps, "p": p,
                "loss": loss, "sharp": sharp,
                "thP": thP, "thA": thA,
                "r": r[:nResid].detach().cpu().tolist(),
                "hfMax": (feTop[0] if feTop else None),
                "hfMin": (feBot[0] if feBot else None),
                "hfTop": feTop[:n], "hfBot": feBot[:n],
                "hlTop": hlt, "hlBot": hlb,
                "gnTop": gnt, "gnBot": gnb, "srTop": srt, "srBot": srb,
                "jt": jt, "jb": jb, "jtN": jtN, "jbN": jbN,
                "paPos": paPos, "paNeg": paNeg, "dimPos": dimPos, "dimNeg": dimNeg,
                "ntkR": ntkR, "ntkH": ntkH, "fhEvT": fhEvT, "fhEvB": fhEvB,
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
                yield {
                    "type": "slq",
                    "sH": slq_density(lambda v: hvpF(th, spec, X, v), p, nProbe, mSLQ, 80, 0x11),
                    "sG": slq_density(lambda v: hvpG(th, spec, X, v), p, nProbe, mSLQ, 80, 0x22),
                    "sS": slq_density(lambda v: hvpS(th, spec, X, v, cS), p, nProbe, mSLQ, 80, 0x33),
                    "sHL": slq_density(lambda v: hvpL(th, spec, X, Y, v), p, nProbe, mSLQ, 80, 0x44),
                }
            eigTick += 1
            done += 1

        th = th - lr * gradL(th, spec, X, Y)[0]

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
        "nsamp": fi("nsamp", 1), "indim": fi("indim", 1), "outdim": fi("outdim", 1),
        "tgt": ff("tgt", 1.0), "neig": fi("neig", 3), "kth": fi("kth", 1),
        "steps": fi("steps", 400), "eigevery": fi("eigevery", 1),
        "slqprobes": fi("slqprobes", 4), "energyp": ff("energyp", 99),
        "seed": fi("seed", 0), "start": fi("start", 0), "inputstd": ff("inputstd", 1.0),
        "ssign": g("ssign", "off"), "qapprox": max(1, fi("qapprox", 25)),
        "qmode": max(1, fi("qmode", 1)), "tset": max(1, fi("tset", 3)),
        "s1": g("s1", "1") == "1", "s2": g("s2", "1") == "1", "s3": g("s3", "1") == "1",
        "s4": g("s4", "1") == "1", "s5": g("s5", "1") == "1", "s6": g("s6", "1") == "1",
        "s7": g("s7", "1") == "1", "s8": g("s8", "1") == "1", "s9": g("s9", "1") == "1",
        "s10": g("s10", "1") == "1", "s11": g("s11", "1") == "1", "s12": g("s12", "0") == "1",
        "gs": g("gson", "1") == "1",
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
        gen = run_stream(_parse_params(q))
        try:
            for msg in gen:
                payload = "data: " + json.dumps(_sanitize(msg)) + "\n\n"
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            gen.close()   # client navigated away / pressed Run again → stop compute
        except Exception as e:  # surface compute errors (OOM, etc.) to the page
            try:
                err = "data: " + json.dumps({"type": "error", "error": str(e)}) + "\n\n"
                self.wfile.write(err.encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass


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
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float64"],
                    help="auto = fp32 on GPU (fast on A6000), fp64 on CPU")
    a = ap.parse_args()

    global DEVICE, DTYPE
    DEVICE = pick_device(a.device)
    if a.dtype == "auto":
        DTYPE = torch.float32 if DEVICE.type == "cuda" else torch.float64
    else:
        DTYPE = torch.float32 if a.dtype == "float32" else torch.float64
    torch.set_grad_enabled(False)
    if DEVICE.type == "cuda":
        torch.cuda.set_device(DEVICE)

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print("EoS / Progressive-Sharpening — GPU backend")
    print(f"  device : {DEVICE}   dtype: {str(DTYPE).replace('torch.', '')}")
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
