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

import threading

import torch
import torch.nn.functional as F

DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cpu")   # default/fallback device (= DEVICE_POOL[0]); set in main()
DEVICE_POOL = []               # every device a run may be assigned to (one GPU each); built in main()
DTYPE = torch.float64          # set in main()
EPS = 1e-3                     # finite-difference step (matches index.html / _preview.py)
PMAX = 10_000_000              # hard cap on parameter count p (matrix-free, so memory is O(p))
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
    eos_lab.models.init_kind_scale / index.html initKindScale. `init` = per-scheme scale/gain/std."""
    fi = max(int(fan_in), 1)
    fo = max(int(fan_out), 1)
    if scheme == "xavier_normal":
        return ("normal", init * math.sqrt(2.0 / (fi + fo)))
    if scheme == "xavier_uniform":
        return ("uniform", init * math.sqrt(6.0 / (fi + fo)))
    if scheme == "mup":
        return ("normal", init / fi if is_readout else init / math.sqrt(fi))
    if scheme == "custom":
        return ("normal", float(init))
    return ("normal", init / math.sqrt(fi))      # "default" — scale/√fan_in (original)


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
        return (z @ p["headw"].t() + p["headb"]).view(N, L)                          # (N, L)


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
    w, V = torch.linalg.eigh(A)
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


# ===================== Lanczos / SLQ =====================
def _randn_vec(p, seed):
    g = torch.Generator(device=_dev())
    g.manual_seed(int(seed) & 0x7FFFFFFF)
    return torch.randn(p, dtype=DTYPE, device=_dev(), generator=g)


def _lanczos_core(hvp, p, m, seed):
    q = _randn_vec(p, seed)
    q = q / q.norm()
    Q = []
    al = []
    be = []
    qp = torch.zeros(p, dtype=DTYPE, device=_dev())
    beta = torch.zeros((), dtype=DTYPE, device=_dev())
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
    Qmat = torch.stack(Q)                     # (k, p) on _dev()

    def ritz(col):
        c = Sv[:, col].to(device=_dev(), dtype=DTYPE)
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


# ===================== datasets =====================
_CIFAR_CACHE = {}


def _find_cifar_dir():
    cands = [CIFAR_DIR,
             os.path.join(DIR, "cifar-10-batches-py"),
             "/home/alexandreduplessis/.torch/cifar-10-batches-py",
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
    elif dataset == "sorting":
        X, Y = load_sort(N, inD, drng)
    elif dataset == "chebyshev":
        X, Y = load_chebyshev(N, P.get("degree", 3))
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
    elif dataset == "sorting":
        inDimE = outDimE = int(P["seqlen"])
    elif dataset == "owt":
        inDimE, outDimE = int(P["seqlen"]), int(P["vocab"])   # block size, vocab
    elif dataset == "chebyshev":
        inDimE = outDimE = 1                                  # scalar x → scalar T_k(x)
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
    if _TL.loss.name == "ce":            # CE: §7/§7a/§8/§4b/§4c use the function NTK (Jᵤ·Jᵤᵀ) and the generic
        s11 = s12 = False                #   residual r=−∂L/∂z (= softmax−onehot), so they're valid. Off for CE:
                                         #   §4d (sign-groups need a scalar residual) and §9 (squared-loss σ₁).
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
    n7 = min(n, max(M, 1))
    n8 = min(n, max(1, p // 2))
    nResid = min(M, 12)
    lr = P["lr"]
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
    start = max(0, min(int(P.get("start", 0)), steps))  # resume: fast-forward GD to here, then stream

    mytok = P.get("_token", 0)                          # per-device token claimed in _sse via acquire_device()
    _devkey = str(_dev())                               # a newer /run on THIS device bumps RUN_TOKEN[_devkey] → we stop

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
    thAccPSD = 0.0; thAccPSD1 = 0.0     # §9b: accumulated 2nd-order PSD term ‖ΔJᵀu₁‖² (≥0), single/multi

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

            needVecs = s4 or s6
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
            if multi_ok and (s7 or s8 or s9 or s10 or s11 or s12 or s13):
                Jc, out_flat = jac_cols(th, X)
                rr = (-N * _TL.loss.resid_cotangent(out, Y, N)).reshape(-1)   # generic residual: Y−f (MSE), onehot−softmax (CE)

            # §7 NTK + function-Hessian tensor SVD
            ntkR = ntkH = fhEvT = fhEvB = None
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
                if s7:                                            # heavy §7 FH-eigenvector projections
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

            # §8 vec(J) onto right singular vecs of the p×(p·dₙ) FH reshape
            g2J = g2Jn = None
            if s8 and multi_ok:
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
            if multi_ok and (s9 or s10 or s11 or s12):
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
                for k in range(n):
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
            thAH = None
            if s14:                       # §9c actual: full loss-Hessian sharpness λmax(∇²L) (reuse §1's if present)
                thAH = sharp if sharp is not None else float(lanczos_extreme_vals(
                    lambda v: hvpL(th, X, Y, v), p, 1, mV, 0x5EED1)[0][0])
            if s12 or s14:
                etaN = lr / max(N, 1); reps_ = max(1, ee)
                if thT0 < 0 or (t - thT0) >= qapprox:               # window start: freeze θ₀, J₀, FH; reset accumulators
                    thT0 = t; thTh0 = th.clone(); thAcc1 = 0.0; thAcc2 = 0.0; thProd3 = 1.0; thProd4 = 1.0
                    thAccPSD = 0.0; thAccPSD1 = 0.0
                    if multi_ok:
                        thJ, _ = jac_cols(th, X)                  # predicted Jacobian J₀ (M, p)
                        thFroz = fh_frozen(th, X, nProbe, min(n, M), 2, mV, M)
                        thBase = float(torch.linalg.eigvalsh(thJ @ thJ.t())[-1])
                    elif not multi:
                        Jc0, _ = gradF(th, X); thJp = Jc0.clone(); thBase = float(Jc0 @ Jc0)
                thP = [None, None, None, None]; thA = [None, None, None, None]
                thPpsd = [None, None, None, None]   # §9b: prediction + accumulated 2nd-order PSD term ‖ΔJᵀu₁‖²
                if multi_ok and thJ is not None and thFroz is not None and bEk_vals is not None:
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
                    # §9b: same predictions + the dropped 2nd-order PSD term Σ‖ΔJᵀu₁‖² (always ≥0 — a sharpening floor)
                    thPpsd[1] = clmp(thBase + thAcc2 + thAccPSD)
                    thPpsd[2] = clmp(thBase*thProd3 + thAccPSD); thPpsd[3] = clmp(thBase*thProd4 + thAccPSD)
                    pnorm = float(rr.norm())                            # col-3 (Eq-22): p = ‖r‖, r̂≈u₁ ⇒
                    thProd3 *= (1 + 2 * etaN * pnorm * fhBil(v1)) ** reps_  #   σ_{t+1}=σ_t[1+2η p·v₁ᵀQ[u₁]v₁]
                    S4 = 0.0; NV = min(max(tset, 1), NV0)              # col-4 (Eq-29): sum over top-|T| modes
                    for vk in range(NV):
                        sgv = math.sqrt(max(float(Kw[vk]), 1e-30)); rho = float(rr @ Vw[:, vk])
                        S4 += (sgv/sgT) * fhBil(gnv(vk)) * rho
                    thProd4 *= (1 + 2 * etaN * S4) ** reps_
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

            # report σ₁ per-sample (÷N) so the theory matches the true sharpness λmax(∇²L) ≈ λmax(GN) = σ₁/N
            thPr = [(x / N if x is not None else None) for x in thP] if thP else None
            thAr = [(x / N if x is not None else None) for x in thA] if thA else None
            thPpsdR = [(x / N if x is not None else None) for x in thPpsd] if thPpsd else None

            yield {
                "type": "step", "t": t, "steps": steps, "p": p,
                "loss": loss, "sharp": sharp,
                "thP": thPr, "thA": thAr, "thPpsd": thPpsdR, "thAH": thAH,
                "r": ([] if r is None else r[:nResid].detach().cpu().tolist()),
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
                    "sH": slq_density(lambda v: hvpF(th, X, v), p, nProbe, mSLQ, 80, 0x11),
                    "sG": slq_density(lambda v: hvpG(th, X, v), p, nProbe, mSLQ, 80, 0x22),
                    "sS": slq_density(lambda v: hvpS(th, X, v, cS), p, nProbe, mSLQ, 80, 0x33),
                    "sHL": slq_density(lambda v: hvpL(th, X, Y, v), p, nProbe, mSLQ, 80, 0x44),
                }
            eigTick += 1
            done += 1

        th = th - lr * gradL(th, X, Y)[0]

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
    cT = cT and not ceLoss          # the Eq-13/21/27/29 σ₁ theory (§9) is derived for squared loss only
    c7a = c7a and not ceLoss        # NTK alignment uses the MSE residual r → MSE only (multi-class CE: off)
    c9c = c9c and not ceLoss        # §9c is the same squared-loss σ₁ recursion → MSE only

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
    t0 = time.time()
    eigTick = 0
    # frozen-window state
    th0 = J0 = gF0 = f0flat = th_sur = feTV0 = feBV0 = None
    thBase = 0.0
    thJ = thFroz = thJp = None
    thAcc1 = thAcc2 = 0.0
    thProd3 = thProd4 = 1.0
    thAccPSD = thAccPSD1 = 0.0      # §9b: accumulated 2nd-order PSD term ‖ΔJᵀu₁‖² (≥0), multi / single

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
            thProd3 = thProd4 = 1.0
            thAccPSD = thAccPSD1 = 0.0
            if cT or c9c:
                if multi:
                    thJ = J0.clone()
                    thFroz = fh_frozen(th0, X, nProbe, min(n, M), 2, mV, M)
                    thBase = float(torch.linalg.eigvalsh(thJ @ thJ.t())[-1])
                else:
                    thJp = gF0.clone()
                    thBase = float(gF0 @ gF0)

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
            if c9c and sharpA is None:        # §9c needs the actual model's λmax(∇²L) even if §eig (cE) is off
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
                J_sur = gF0 + (hvpF(th0, X, dth) if kind == "quad" else zero_p)
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

            # ---- §9 theory σ₁ (unchanged frozen-Q propagation) vs the SURROGATE's measured σ₁ ----
            # §9b adds the dropped 2nd-order PSD term; §9c compares against the actual model's full
            # loss-Hessian sharpness λmax(∇²L) (=sharpA) instead of the surrogate's Gauss-Newton edge.
            thP = thA = thPpsd = None
            thAH = sharpA if c9c else None
            if cT or c9c:
                etaN = lr / max(N, 1)
                reps_ = max(1, ee)
                thP = [None, None, None, None]
                thA = [None, None, None, None]
                thPpsd = [None, None, None, None]
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
                    sigSur = float(torch.linalg.eigvalsh(Jq @ Jq.t())[-1])   # surrogate σ₁ (measured)
                    thA[1] = thA[2] = thA[3] = sigSur
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
                    thP[3] = clmp(thBase * thProd4)
                    thPpsd[1] = clmp(thBase + thAcc2 + thAccPSD)          # §9b: + 2nd-order PSD term Σ‖ΔJᵀu₁‖²
                    thPpsd[2] = clmp(thBase * thProd3 + thAccPSD)
                    thPpsd[3] = clmp(thBase * thProd4 + thAccPSD)
                    pnorm = float(rr.norm())                          # col-3 (Eq-22): p=‖r‖, σ_{t+1}=σ_t[1+2η p·v₁ᵀQ[u₁]v₁]
                    thProd3 *= (1 + 2 * etaN * pnorm * fhBil(v1)) ** reps_
                    S4 = 0.0
                    NV = min(max(tset, 1), NV0)
                    for vk in range(NV):
                        sgv = math.sqrt(max(float(Kw[vk]), 1e-30))
                        rho = float(rr @ Vw[:, vk])
                        S4 += (sgv / sgT) * fhBil(gnv(vk)) * rho
                    thProd4 *= (1 + 2 * etaN * S4) ** reps_
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

            # report σ₁ per-sample (÷N) so the theory matches the true sharpness scale (σ₁/N)
            thPr = [(x / N if x is not None else None) for x in thP] if thP else None
            thAr = [(x / N if x is not None else None) for x in thA] if thA else None
            thPpsdR = [(x / N if x is not None else None) for x in thPpsd] if thPpsd else None

            yield {
                "type": "step", "t": t, "steps": steps, "p": p, "kind": kind,
                "lossA": lossA if cL else None, "lossS": lossS if cL else None,
                "rmeanA": rmeanA if cR else None, "rstdA": rstdA if cR else None,
                "rmeanS": rmeanS if cR else None, "rstdS": rstdS if cR else None,
                "eigA": eigA, "eigS": eigS, "sharpA": sharpA, "sharpS": sharpS,
                "thP": thPr, "thA": thAr, "thPpsd": thPpsdR, "thAH": thAH,
                "j4tA": j4tA, "j4bA": j4bA, "j4tS": j4tS, "j4bS": j4bS,
                "d4tA": d4tA, "d4bA": d4bA, "d4tS": d4tS, "d4bS": d4bS,
                "ntkR": ntkR, "ntkH": ntkH,
                "sps": (t - start + 1) / max(time.time() - t0, 1e-9),
            }
            if cH and eigTick % slqStride == 0:
                yield _resid_hist(rA, rS)
            eigTick += 1

        # advance the actual model AND the frozen-window surrogate by one GD step EVERY iteration, so the
        # surrogate stays in lockstep with the actual model regardless of eigevery / qapprox alignment
        # (the surrogate GD is on the frozen-θ₀ MSE Taylor model; Q,J₀ are frozen at the window start).
        th = th - lr * gradL(th, X, Y)[0]
        if th0 is not None:
            dths = th_sur - th0
            if kind == "quad":
                QDs = jac_hvp(th0, X, dths)
                fss = f0flat + (J0 @ dths) + 0.5 * (QDs @ dths)
                Jqs = J0 + QDs
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
        "slqprobes": fi("slqprobes", 4), "energyp": ff("energyp", 99),
        "seed": fi("seed", 0), "start": fi("start", 0), "inputstd": ff("inputstd", 1.0),
        "ssign": g("ssign", "off"), "qapprox": max(1, fi("qapprox", 25)),
        "qmode": max(1, fi("qmode", 1)), "tset": max(1, fi("tset", 3)),
        "dataset": g("dataset", "synthetic"), "arch": g("arch", "mlp"), "loss": g("loss", "mse"),
        "chmul": ff("chmul", 0.25), "nlayer": fi("nlayer", 2), "nhead": fi("nhead", 4),
        "dmodel": fi("dmodel", 64), "seqlen": fi("seqlen", 16), "vocab": fi("vocab", 50257),
        "degree": fi("degree", 3),       # Chebyshev polynomial degree (chebyshev dataset)
        "initscheme": g("initscheme", "default"),
        "surrogate": g("surrogate", "quad"), "mode": g("mode", "run"),
        "s1": g("s1", "1") == "1", "s2": g("s2", "1") == "1", "s3": g("s3", "1") == "1",
        "s4": g("s4", "1") == "1", "s5": g("s5", "1") == "1", "s6": g("s6", "1") == "1",
        "s7": g("s7", "1") == "1", "s8": g("s8", "1") == "1", "s9": g("s9", "1") == "1",
        "s10": g("s10", "1") == "1", "s11": g("s11", "1") == "1", "s12": g("s12", "0") == "1",
        "s13": g("s13", "1") == "1",
        "s14": g("s14", "0") == "1",     # §9c: σ₁ predictions vs full loss-Hessian sharpness
        "gs": g("gson", "1") == "1",
        # surrogate-section panel toggles (loss · resid mean/std · top-n eig · histogram · theory · §4 · §4d)
        "c1": g("c1", "1") == "1", "c2": g("c2", "1") == "1", "c3": g("c3", "1") == "1",
        "c4": g("c4", "1") == "1", "c5": g("c5", "0") == "1", "c6": g("c6", "1") == "1",
        "c7": g("c7", "1") == "1", "c8": g("c8", "1") == "1", "c9": g("c9", "0") == "1",
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
