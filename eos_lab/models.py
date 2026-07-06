"""
Models + losses — one flat parameter vector θ, functional forward, autograd gradients/HVPs.

MIRRORS:  server.py  (build_spec, fwd, Model/MlpModel/AutogradModel/CnnModel/Vgg11Model/GptModel,
                       MSELoss/CELoss, build_loss, build_model, gradF/gradL/gradW, hvpF/L/S/G)
          ↔ index.html  (the same architectures, computed in JS for the MLP)

Design (kept deliberately uniform and interpretable):
  * Every model exposes  forward(theta, X) -> out (N, oc)  with ALL parameters packed in one flat
    tensor `theta` (shape (p,)).  This is what makes the whole matrix-free eigen-suite in linalg.py
    architecture-agnostic.
  * Gradients and Hessian–vector products come from torch autograd (exact double-backward), so there
    is no finite-difference machinery to read.  (server.py uses finite differences only for the MLP to
    stay bit-identical to the browser; the GD *trajectory* here still matches because the forward map,
    the init, and the data are identical — only the curvature probes differ, and autograd is exact.)
  * The MLP init uses mulberry32 (browser-identical); conv/transformer inits use torch's RNG.
"""
import math
import torch
import torch.nn.functional as F

from .rng import mulberry32, u32, gauss


# --------------------------------------------------------------------------- init schemes
def init_kind_scale(scheme, init, fan_in, fan_out, is_readout):
    """How to draw a weight tensor under `scheme`. Returns ('normal', std) or ('uniform', half-range).
    The named theoretical schemes use their CANONICAL variance and IGNORE the `init_scale` knob:
      mup           → hidden std = 1/√fan_in, readout std = 1/fan_in   (canonical muP base init)
      xavier_normal → std = √(2/(fan_in+fan_out))                       (Glorot normal, gain 1)
      xavier_uniform→ U(±√(6/(fan_in+fan_out)))                         (Glorot uniform, gain 1)
    `init_scale` only applies to the two free-magnitude schemes:
      default       → std = init/√fan_in   (scale)
      custom        → std = init            (the Gaussian std directly)
    MIRRORS server.init_kind_scale / index.html initKindScale."""
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


# --------------------------------------------------------------------------- activations
def actf(name, z):
    if name == "tanh":
        return torch.tanh(z)
    if name == "elu":
        return torch.where(z > 0, z, torch.expm1(z))
    if name == "quadratic":
        return z * z
    return z


# --------------------------------------------------------------------------- MLP
def build_spec(in_dim, width, depth, act, use_bias, out_dim):
    """Layer layout: list of (din, dout, act, w_off, b_off) + total param count p.
    Hidden layers use `act`; the final layer is linear.  b_off = -1 when bias is off.
    MIRRORS server.build_spec / index.html buildSpec exactly (same offsets ⇒ same θ packing)."""
    spec, off, din = [], 0, in_dim
    for _ in range(depth):
        dout = width
        w_off = off
        off += din * dout
        b_off = off if use_bias else -1
        if use_bias:
            off += dout
        spec.append((din, dout, act, w_off, b_off))
        din = dout
    w_off = off
    off += din * out_dim
    b_off = off if use_bias else -1
    if use_bias:
        off += out_dim
    spec.append((din, out_dim, "linear", w_off, b_off))
    return spec, off


def mlp_forward(theta, spec, X):
    """Functional MLP forward from the flat θ (differentiable w.r.t. θ). MIRRORS server.fwd."""
    a = X
    for (din, dout, act, w_off, b_off) in spec:
        W = theta[w_off:w_off + din * dout].view(din, dout)
        z = a @ W
        if b_off >= 0:
            z = z + theta[b_off:b_off + dout]
        a = actf(act, z)
    return a


class Model:
    """Common interface. Subclasses set p / oc / in_shape and implement forward + init_theta."""
    p = 0
    oc = 1
    in_shape = None
    init_scheme = "default"        # set by build_model from cfg.initscheme

    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype

    def forward(self, theta, X):
        raise NotImplementedError

    def init_theta(self, seed, init_scale):
        raise NotImplementedError


class MlpModel(Model):
    """Dense MLP. Seeded init via mulberry32 ⇒ same θ₀ (and thus same GD trajectory) as the browser."""
    def __init__(self, in_dim, out_dim, P, device, dtype):
        super().__init__(device, dtype)
        self.spec, self.p = build_spec(in_dim, P["width"], P["depth"], P["act"],
                                       P["bias"] == "1", out_dim)
        self.in_shape, self.oc = (in_dim,), out_dim

    def forward(self, theta, X):
        return mlp_forward(theta, self.spec, X)

    def init_theta(self, seed, init_scale):
        rng = mulberry32(u32(seed))
        th = [0.0] * self.p
        nl = len(self.spec)
        for li, (din, dout, act, w_off, b_off) in enumerate(self.spec):
            kind, sc = init_kind_scale(self.init_scheme, init_scale, din, dout, li == nl - 1)
            for i in range(din * dout):
                th[w_off + i] = (gauss(rng) * sc) if kind == "normal" else ((rng() * 2.0 - 1.0) * sc)
        return torch.tensor(th, dtype=self.dtype, device=self.device)


# --------------------------------------------------------------------------- autograd nets
class AutogradModel(Model):
    """Base for conv / transformer nets. Parameters are registered with _add() into one flat θ;
    `_net(param_dict, X)` defines the forward. Seeded init uses torch's RNG. MIRRORS server.AutogradModel."""
    def __init__(self, device, dtype):
        super().__init__(device, dtype)
        self._specs = []   # (name, shape, numel, offset, fan_in, init∈{'n','0','1'})
        self.p = 0

    def _add(self, name, shape, fan_in=None, init="n"):
        numel = 1
        for s in shape:
            numel *= s
        self._specs.append((name, tuple(shape), numel, self.p, fan_in, init))
        self.p += numel

    def _unflatten(self, theta):
        return {name: theta[off:off + numel].view(*shape)
                for name, shape, numel, off, fan_in, init in self._specs}

    def init_theta(self, seed, init_scale):
        g = torch.Generator(device="cpu")
        g.manual_seed((int(seed) & 0x7FFFFFFF) or 1)
        readout_off = max((off for _, _, _, off, _, ini in self._specs if ini == "n"), default=-1)
        parts = []
        for name, shape, numel, off, fan_in, init in self._specs:
            if init == "0":
                parts.append(torch.zeros(numel))
            elif init == "1":
                parts.append(torch.ones(numel))
            else:
                fi = fan_in or numel
                fo = max(1, numel // fi)               # fan_out = numel/fan_in for weight tensors
                kind, sc = init_kind_scale(self.init_scheme, init_scale, fi, fo, off == readout_off)
                draw = (torch.randn(numel, generator=g) if kind == "normal"
                        else (torch.rand(numel, generator=g) * 2.0 - 1.0))
                parts.append(draw * sc)
        return torch.cat(parts).to(dtype=self.dtype, device=self.device)

    def forward(self, theta, X):
        return self._net(self._unflatten(theta), X)

    def _net(self, p, X):
        raise NotImplementedError


class CnnModel(AutogradModel):
    """3 conv blocks (conv→act→avg-pool) + global-avg-pool + linear head. MIRRORS server.CnnModel."""
    def __init__(self, in_dim, out_dim, P, device, dtype):
        super().__init__(device, dtype)
        assert in_dim == 3 * 32 * 32, "CNN expects CIFAR-shaped input (3×32×32)"
        self.in_shape, self.oc = (3, 32, 32), out_dim
        self.act = P.get("act", "tanh")
        m = max(0.1, float(P.get("chmul", 1.0)))
        cin = 3
        self._convs = []
        for j, c in enumerate(max(1, int(round(c * m))) for c in (32, 64, 64)):
            self._add(f"c{j}w", (c, cin, 3, 3), fan_in=cin * 9)
            self._add(f"c{j}b", (c,), init="0")
            self._convs.append((f"c{j}w", f"c{j}b"))
            cin = c
        self._add("fcw", (out_dim, cin), fan_in=cin)
        self._add("fcb", (out_dim,), init="0")

    def _net(self, p, X):
        h = X.view(X.shape[0], 3, 32, 32)
        for wn, bn in self._convs:
            h = F.avg_pool2d(actf(self.act, F.conv2d(h, p[wn], p[bn], padding=1)), 2)
        return h.mean(dim=(2, 3)) @ p["fcw"].t() + p["fcb"]


class Vgg11Model(AutogradModel):
    """VGG11 stack (CIFAR-adapted, no BatchNorm, avg-pool) scaled by chmul + linear head.
    chmul=1 → ~9.4M params; chmul=0.25 → ~0.6M. MIRRORS server.Vgg11Model."""
    CFG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self, in_dim, out_dim, P, device, dtype):
        super().__init__(device, dtype)
        assert in_dim == 3 * 32 * 32, "VGG11 expects CIFAR-shaped input (3×32×32)"
        self.in_shape, self.oc = (3, 32, 32), out_dim
        self.act = P.get("act", "tanh")
        m = max(0.05, float(P.get("chmul", 0.25)))
        cin, j = 3, 0
        self._layers = []
        for v in self.CFG:
            if v == "M":
                self._layers.append(("M",))
            else:
                c = max(1, int(round(v * m)))
                self._add(f"c{j}w", (c, cin, 3, 3), fan_in=cin * 9)
                self._add(f"c{j}b", (c,), init="0")
                self._layers.append(("C", f"c{j}w", f"c{j}b"))
                cin, j = c, j + 1
        self._add("fcw", (out_dim, cin), fan_in=cin)
        self._add("fcb", (out_dim,), init="0")

    def _net(self, p, X):
        h = X.view(X.shape[0], 3, 32, 32)
        for L in self._layers:
            h = F.avg_pool2d(h, 2) if L[0] == "M" else actf(self.act, F.conv2d(h, p[L[1]], p[L[2]], padding=1))
        return h.view(h.shape[0], -1) @ p["fcw"].t() + p["fcb"]


class GptModel(AutogradModel):
    """Mini-GPT for the sorting task: scalar→d_model embed + learned positional, nlayer pre-norm
    blocks (full self-attention + MLP), linear head → one scalar per position. MIRRORS server.GptModel."""
    def __init__(self, in_dim, out_dim, P, device, dtype):
        super().__init__(device, dtype)
        self.L, self.oc, self.in_shape = in_dim, out_dim, (in_dim,)
        self.pool = (P.get("dataset") == "ksparse")     # k-sparse parity: mean-pool the per-position head → ONE scalar (oc=1)
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
        z = X.view(N, L, 1) @ p["emb_w"].t() + p["emb_b"] + p["pos"].unsqueeze(0)
        for i in range(self.nl):
            q = f"b{i}_"
            a = self._ln(z, p[q + "ln1g"], p[q + "ln1b"])
            qkv = a @ p[q + "qkvw"].t() + p[q + "qkvb"]
            qq, kk, vv = qkv.split(d, dim=2)
            qq = qq.view(N, L, h, dh).transpose(1, 2)
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
    """Mini-GPT *language model* for OpenWebText: token-embedding lookup + learned positional, nlayer
    pre-norm blocks (full causal self-attention + MLP), final LN, weight-TIED head (logits = h·Wteᵀ),
    so the (vocab×d) embedding doubles as the output projection. Input X is (N, block) int token ids;
    output is (N, block, vocab) next-token logits (cross-entropy). MIRRORS server.GptLMModel.

    Differs from GptModel (the sorting regressor) only in the embedding/head: scalar emb+head → token
    table + tied head, and the attention is causally masked (a real autoregressive LM)."""
    def __init__(self, block, vocab, P, device, dtype):
        super().__init__(device, dtype)
        self.L, self.V, self.in_shape = block, vocab, (block,)
        self.oc = vocab
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
        Xl = X.long()
        z = p["wte"][Xl] + p["wpe"][:T].unsqueeze(0)            # (N,T,d)
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
        return z @ p["wte"].t()                                 # tied head → (N,T,vocab) logits


class DiagLinModel(Model):
    """Diagonal linear network (Alternating Gradient Flows, arXiv:2506.06489): β = u⊙v per output channel,
    f_c(x) = Σ_i u_{c,i} v_{c,i} x_i. From small init the coordinates activate one at a time → multi-step
    saddle-to-saddle. p = 2·oc·d. MIRRORS server.DiagLinModel (torch.Generator init ⇒ same θ₀ as the server)."""
    def __init__(self, in_dim, out_dim, P, device, dtype):
        super().__init__(device, dtype)
        self.d = max(1, int(in_dim)); self.oc = max(1, int(out_dim)); self.in_shape = (self.d,)
        self.p = 2 * self.oc * self.d

    def forward(self, theta, X):
        n = self.oc * self.d
        u = theta[:n].view(self.oc, self.d); v = theta[n:].view(self.oc, self.d)
        return X @ (u * v).t()                                  # (N, oc) ; β = u⊙v

    def init_theta(self, seed, init_scale):
        g = torch.Generator(device="cpu"); g.manual_seed((int(seed) & 0x7FFFFFFF) or 1)
        kind, sc = init_kind_scale(self.init_scheme, init_scale, self.d, self.oc, False)
        draw = (torch.randn(self.p, generator=g) if kind == "normal" else (torch.rand(self.p, generator=g) * 2.0 - 1.0))
        return (draw * sc).to(dtype=self.dtype, device=self.device)


def build_model(arch, in_dim, out_dim, P, device, dtype):
    """Factory. MIRRORS server.build_model."""
    if arch == "mlp":
        m = MlpModel(in_dim, out_dim, P, device, dtype)
    elif arch == "cnn":
        m = CnnModel(in_dim, out_dim, P, device, dtype)
    elif arch == "vgg11":
        m = Vgg11Model(in_dim, out_dim, P, device, dtype)
    elif arch == "gpt":
        # owt → token-LM GPT (CE over vocab); sorting → scalar-regression GPT (MSE).
        m = (GptLMModel(in_dim, out_dim, P, device, dtype) if P.get("dataset") == "owt"
             else GptModel(in_dim, out_dim, P, device, dtype))
    elif arch == "diaglin":
        m = DiagLinModel(in_dim, out_dim, P, device, dtype)
    else:
        raise ValueError(f"unknown arch '{arch}'")
    m.init_scheme = P.get("initscheme", "default")     # weight-init scheme (default/mup/xavier_*/custom)
    return m


# --------------------------------------------------------------------------- losses
class MSELoss:
    """Squared error. Handles two shapes uniformly (sum over the last/output axis):
      * regression / one-hot classification — out (N, C),    Y (N, C) float       → ÷N
      * next-token LM (OpenWebText)          — out (N, T, V), Y token ids (N, T)   → one-hot, ÷N·T
    so the language model can be trained with MSE (regress the logits toward the one-hot next token).
    The per-token normalisation mirrors CELoss, so a given learning rate sits at a comparable scale
    whether MSE or CE is selected on OWT."""
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

    def resid_cotangent(self, out, Y, N):   # ∂L/∂out, so ∇_θL = J^T · this
        return (out - self._target(out, Y)) / self._tokens(out, N)

    def gn_apply(self, out, Jv, N):          # output-space Gauss–Newton metric A = I/tokens
        return Jv / self._tokens(out, N)


class CELoss:
    """Softmax cross-entropy. Handles two shapes uniformly (softmax over the last axis):
      * classification  — out (N, C),     Y one-hot (N, C)         → normalised per sample  (÷N)
      * next-token LM    — out (N, T, V),  Y token ids long (N, T)  → normalised per token   (÷N·T)
    so the owt language model reports the standard per-token CE (≈ln V at init)."""
    name = "ce"

    @staticmethod
    def _tokens(out, N):
        return N * out.shape[1] if out.dim() == 3 else N      # per-token (LM) vs per-sample

    @staticmethod
    def _onehot(out, Y):
        if Y.dtype == torch.long:                             # LM integer targets → one-hot on the fly
            return torch.zeros_like(out).scatter_(-1, Y.unsqueeze(-1), 1.0)
        return Y

    def value(self, out, Y, N):
        logp = torch.log_softmax(out, dim=-1)
        if Y.dtype == torch.long:
            nll = -logp.gather(-1, Y.unsqueeze(-1)).squeeze(-1)
            return nll.sum() / self._tokens(out, N)
        return -(Y * logp).sum() / self._tokens(out, N)

    def resid_cotangent(self, out, Y, N):    # ∂L/∂out
        return (torch.softmax(out, dim=-1) - self._onehot(out, Y)) / self._tokens(out, N)

    def gn_apply(self, out, Jv, N):          # A = (diag(p) − ppᵀ)/tokens per scalar output
        p = torch.softmax(out, dim=-1)
        return (p * Jv - p * (p * Jv).sum(dim=-1, keepdim=True)) / self._tokens(out, N)


def build_loss(name):
    return CELoss() if name == "ce" else MSELoss()


# --------------------------------------------------------------------------- grads + HVPs (autograd)
# These are the matrix-free primitives every diagnostic is built from. Each returns a flat (p,) vector.
def grad_sum_f(model, theta, X):
    """(∇_θ Σ_i f(x_i),  out).  J = ∇_θ Σf is the §1/§4 'function gradient'. MIRRORS server.gradF."""
    with torch.enable_grad():
        th = theta.detach().requires_grad_(True)
        out = model.forward(th, X)
        g, = torch.autograd.grad(out.sum(), th)
    return g.detach(), out.detach()


def grad_loss(model, loss, theta, X, Y):
    """(∇_θ L,  out).  Full-batch loss gradient — this is the GD step direction. MIRRORS server.gradL."""
    N = X.shape[0]
    with torch.enable_grad():
        th = theta.detach().requires_grad_(True)
        out = model.forward(th, X)
        L = loss.value(out, Y, N)
        g, = torch.autograd.grad(L, th)
    return g.detach(), out.detach()


def grad_weighted(model, theta, X, c):
    """(∇_θ Σ(c⊙f)).  Backward with an arbitrary output cotangent c (N,oc). MIRRORS server.gradW."""
    with torch.enable_grad():
        th = theta.detach().requires_grad_(True)
        out = model.forward(th, X)
        g, = torch.autograd.grad((out * c).sum(), th)
    return g.detach()


def matpow_sym(M, power, rel=1e-6):
    """Mᵖ for a symmetric PSD matrix via eigendecomposition, eigenvalues floored at rel·λmax so a NEGATIVE
    power of a rank-deficient / singular M is finite (near-null directions are clamped, not blown up).
    MIRRORS server._matpow_sym EXACTLY (plain torch.linalg.eigh — no safe wrapper — for bit-parity)."""
    lam, V = torch.linalg.eigh(M)
    lmax = float(lam[-1]) if lam.numel() else 0.0
    lam = lam.clamp_min(max(rel * lmax, 1e-30))
    return (V * lam.pow(power)) @ V.t()


def muon_whiten(model, v):
    """Muon / spectral-GD preconditioner applied to v: per weight MATRIX the block is
    (GGᵀ)^(−¼) ⊗ (GᵀG)^(−¼) with G = that layer's slice of v — i.e. the two-sided whitening
    (GGᵀ)^(−¼)·G·(GᵀG)^(−¼), which equals UVᵀ (the polar factor) for full-rank G. Biases / non-matrix
    params → sign(v). MLP only (reads model.spec); returns sign(v) for models without a spec.
    MIRRORS server._muon_whiten EXACTLY (the matpow form, NOT the old SVD U@Vh)."""
    d = torch.sign(v)                            # default for biases / non-matrix params
    spec = getattr(model, "spec", None)
    if spec is not None:                         # MLP: layer ℓ weight is a din×dout contiguous block of θ
        for (din, dout, act, w_off, b_off) in spec:
            W = v[w_off:w_off + din * dout].view(din, dout)
            A = matpow_sym(W @ W.t(), -0.25)     # (GGᵀ)^(−¼)   din×din
            B = matpow_sym(W.t() @ W, -0.25)     # (GᵀG)^(−¼)   dout×dout
            d[w_off:w_off + din * dout] = (A @ W @ B).reshape(-1)
    return d


def gn_precond(J, r, k=50, rel=1e-6):
    """Min-norm Gauss-Newton direction  Jᵀ(JJᵀ)⁻¹r  (drives f→Y). For M≤k a single damped SPD solve
    (JJᵀ+λI)⁻¹r (full inverse; cheap — no eigendecomposition). For M>k the top-k eigenpairs of JJᵀ
    (rank-k truncated pseudo-inverse). MIRRORS server._gn_precond EXACTLY."""
    M = J.shape[0]
    K = J @ J.t()                                # M×M NTK
    if M <= k:                                   # full inverse via a damped Cholesky solve (avoids the per-call eigh cost)
        ridge = rel * K.diagonal().mean() + 1e-30
        return J.t() @ torch.linalg.solve(K + ridge * torch.eye(M, dtype=K.dtype, device=K.device), r)
    lam, U = torch.linalg.eigh(K)                # M>k: eigenvalues ascending
    lam_k = lam[-k:]; U_k = U[:, -k:]            # top-k directions
    inv = 1.0 / lam_k.clamp_min(max(rel * float(lam_k[-1]), 1e-30))
    return J.t() @ (U_k @ (inv * (U_k.t() @ r)))  # Jᵀ Σ (1/λ_i)(u_iᵀr) u_i


def opt_dir(model, g, opt, J=None, r=None):
    """Update DIRECTION d (NO momentum, NO weight decay); the actual θ step is θ ← θ − lr·d. MIRRORS
    server._opt_dir (4-way).
      gd          → g = ∇L                       (plain full-batch gradient descent)
      sign        → sign(g)                      (signed GD, elementwise)
      spectral    → Muon: per weight matrix (GGᵀ)^(−¼)·G·(GᵀG)^(−¼) (= UVᵀ full-rank); biases → sign(g)
      gaussnewton → −Jᵀ(JJᵀ)⁻¹r  (min-norm GN; θ←θ+lr·Jᵀ(JJᵀ)⁻¹r drives f→Y; top-50 truncated). Needs J,r.
    Only the trajectory changes here; the preconditioned prediction dynamics use precond_flow."""
    if opt == "sign":
        return torch.sign(g)
    if opt == "spectral":
        return muon_whiten(model, g)
    if opt == "gaussnewton":
        return -gn_precond(J, r)
    return g                                     # gd (default)


def precond_flow(model, opt, J, r, N, lr):
    """(g, scale) for the preconditioned prediction dynamics: one optimizer step moves θ by Δθ = scale·g, so
    ṙ = −scale·J·g,  J̇ = scale·Q·g,  the self-trajectory Δθ = scale·g, and the 2nd-order residual curvature
    = ½·scale²·(gᵀQg). g is a WELL-SCALED direction; the per-optimizer step size lives in `scale`. This
    matches the ACTUAL step θ←θ−lr·opt_dir(∇L,opt). gd ⇒ (Jᵀr, lr/N) reproduces GD byte-for-byte.
    MIRRORS server._precond_flow EXACTLY.
      gd → (Jᵀr, lr/N)   sign → (sign(Jᵀr), lr)   spectral → (Muon-whiten(Jᵀr), lr)   gaussnewton → (Jᵀ(JJᵀ)⁻¹r, lr)."""
    Jr = J.t() @ r
    if opt == "sign":
        return torch.sign(Jr), lr
    if opt == "spectral":
        return muon_whiten(model, Jr), lr
    if opt == "gaussnewton":
        return gn_precond(J, r), lr
    return Jr, lr / max(N, 1)                     # gd (default): (Jᵀr, η/N)


def _hvp_scalar(model, theta, X, scalar_fn, v):
    """Generic exact HVP of a scalar(θ): (∇² scalar) · v via double-backward (Pearlmutter).

    NOTE: this is EXACT autograd, whereas server.py's MlpModel uses *finite-difference* HVPs here
    (kept for byte-parity with the browser). So §1/§2/§3 sharpness & H/H_L/G/S eigenvalues match the
    server to ~1e-6 in fp64 but diverge ~1% on fp32/GPU. The multi-sample primitives (jac_hvp/hvp_G2 →
    §4/§7/§8/§15/§16/§17) DO mirror server's FD definitions, so those match to 1e-5..1e-13."""
    with torch.enable_grad():
        th = theta.detach().requires_grad_(True)
        out = model.forward(th, X)
        g, = torch.autograd.grad(scalar_fn(out), th, create_graph=True)
        Hv, = torch.autograd.grad((g * v).sum(), th)
    return Hv.detach()


def hvp_F(model, theta, X, v):
    """Function Hessian (∇² Σf)·v. MIRRORS server.hvpF."""
    return _hvp_scalar(model, theta, X, lambda out: out.sum(), v)


def hvp_L(model, loss, theta, X, Y, v):
    """Loss Hessian (∇²L)·v — its top eigenvalue is the *sharpness*. MIRRORS server.hvpL."""
    N = X.shape[0]
    return _hvp_scalar(model, theta, X, lambda out: loss.value(out, Y, N), v)


def hvp_S(model, theta, X, c, v):
    """Residual term (Σ_a c_a ∇²f_a)·v, c = ∂L/∂out (N,oc). loss-Hessian = G + S. MIRRORS server.hvpS."""
    cc = c.detach()
    return _hvp_scalar(model, theta, X, lambda out: (out * cc).sum(), v)


def hvp_G(model, loss, theta, X, v):
    """Gauss–Newton (Jᵀ A J)·v with the loss-aware output metric A. PSD. MIRRORS server.hvpG."""
    N = X.shape[0]
    th = theta.detach()

    def fwd(t):
        return model.forward(t, X)

    with torch.enable_grad():
        out, Jv = torch.func.jvp(fwd, (th,), (v,))          # Jv = J·v in output space
        AJv = loss.gn_apply(out.detach(), Jv.detach(), N)   # A·(J·v)
        thl = th.detach().requires_grad_(True)
        o2 = model.forward(thl, X)
        Gv, = torch.autograd.grad((o2 * AJv).sum(), thl)    # Jᵀ·(A J v)
    return Gv.detach()


# --------------------------------------------------------------------------- multi-sample primitives
# These power the §4 / §7 / §8 / §4b–§4d / §9 projection panels.  M = N·oc is the number of scalar
# outputs;  Jc (M,p) is the per-output Jacobian (row a = ∇f_a).  jac_hvp / hvpG2 mirror server.py's
# *finite-difference* definitions (EPS=1e-3) exactly, so these sections match server.py numerically.
FD_EPS = 1e-3


def jac_cols(model, theta, X):
    """Per-output Jacobian J (M,p) (row a = ∇f_a) + flat out (M,). Vectorized vjp. MIRRORS server.jac_cols."""
    th = theta.detach()

    def f(t):
        return model.forward(t, X).reshape(-1)

    out = f(th).detach()
    M = out.shape[0]
    chunk = max(16, min(M, 8_000_000 // max(model.p, 1)))   # bound the vmapped backward's memory
    with torch.enable_grad():
        try:
            J = torch.func.jacrev(f, chunk_size=chunk)(th)
        except TypeError:                                   # older torch without chunk_size
            J = torch.func.jacrev(f)(th)
    return J.detach(), out


def grad_out(model, theta, X, a):
    """∇f_a — gradient of a single (flattened) output index a. MIRRORS server.grad_out."""
    with torch.enable_grad():
        th = theta.detach().requires_grad_(True)
        out = model.forward(th, X).reshape(-1)
        g, = torch.autograd.grad(out[a], th)
    return g.detach()


def jac_hvp(model, theta, X, z, eps=FD_EPS):
    """{∇²f_a · z}_a — central difference of jac_cols. Shape (M, p). MIRRORS server.jac_hvp."""
    cp, _ = jac_cols(model, theta + eps * z, X)
    cm, _ = jac_cols(model, theta - eps * z, X)
    return (cp - cm) / (2 * eps)


def hvp_G2(model, theta, X, v, eps=FD_EPS):
    """G2 = Σ_a (∇²f_a)² applied to v. MIRRORS server.hvpG2."""
    u = jac_hvp(model, theta, X, v, eps)
    out = torch.zeros(theta.shape[0], dtype=theta.dtype, device=theta.device)
    for a in range(u.shape[0]):
        out += (grad_out(model, theta + eps * u[a], X, a)
                - grad_out(model, theta - eps * u[a], X, a)) / (2 * eps)
    return out
