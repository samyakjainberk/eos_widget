"""
Run configuration + presets.

MIRRORS:  index.html  (the control panel + PRESETS dropdown)  and  server.py _parse_params.

`Config` is a typed dataclass of every knob the widget exposes. `.P()` returns the plain dict the
numerics consume (string-valued flags like bias/fixedx kept identical to server.py's P, so the two
stay drop-in compatible). PRESETS mirrors the index.html PRESETS object 1:1.
"""
from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # ---- dataset & architecture (mirror index.html 'Dataset & architecture' panel) ----
    dataset: str = "synthetic"      # synthetic | cifar10 | sorting | owt | chebyshev
    arch: str = "mlp"               # mlp | cnn | vgg11 | gpt
    loss: str = "mse"               # mse | ce
    chmul: float = 0.25             # CNN/VGG channel multiplier
    dmodel: int = 64                # GPT d_model
    nhead: int = 4                  # GPT heads
    nlayer: int = 2                 # GPT blocks
    seqlen: int = 16                # sorting sequence length / owt block size (tokens per sequence)
    vocab: int = 50257              # owt vocabulary (GPT-2 BPE) — token-embedding/head size
    degree: int = 3                 # chebyshev: degree of the target Chebyshev polynomial T_k

    # ---- model (mirror 'Model' panel) ----
    depth: int = 4
    width: int = 100
    act: str = "linear"             # linear | tanh | elu
    bias: int = 0                   # 0/1
    fixedx: int = 1                 # 1 = fixed input x=1, 0 = iid Gaussian
    inputstd: float = 1.0
    ssign: str = "off"              # off | pos | neg  (initial residual sign)
    indim: int = 10
    outdim: int = 1

    # ---- data & training (mirror 'Data & training' panel) ----
    nsamp: int = 1
    batch: int = 0                  # minibatch size for the GD step + per-step diagnostics
                                    # (0 ⇒ full batch = nsamp). Only the owt dataset uses it today.
    tgt: float = 1.0
    lr: float = 0.3
    init: float = 0.1               # init magnitude; meaning depends on initscheme (below)
    initscheme: str = "default"     # default(=scale/√fan_in) | mup | xavier_normal | xavier_uniform |
                                    #   custom(=Gaussian std = init). `init` is the scale/gain/std per scheme.
    steps: int = 400
    seed: int = 0

    # ---- analysis (mirror 'Analysis & simulation' panel) ----
    neig: int = 3                   # top/bottom-n eigenpairs
    kth: int = 1
    eigevery: int = 1               # compute diagnostics every k GD steps
    slqprobes: int = 4
    energyp: float = 99.0           # §6 subspace energy %
    gson: int = 1                   # compute G & S spectra in §2/§3
    qapprox: int = 25              # §9 frozen-Q window length (steps)
    qmode: int = 1                 # §9 col-3 mode index — legacy (Eq-22 uses no mode)
    tset: int = 3                  # §9 Eq-29 number of top modes |T|

    # ---- which sections to compute ----
    # Section ↔ widget toggle map:  s1 §1 · s2 §2(top) · s3 §3(bottom) · s4 §4(J→H eigvecs) ·
    #   s5 §5(SLQ) · s6 §6(rotation) · s7 §7(multi-sample NTK/FH SVD) · s8 §8(FH-reshape SVD) ·
    #   s9 §4b(J·r→Q[u₁]) · s10 §4c(Q[u₁]·Jr→GN) · s11 §4d(per-sign-group) · s12 §9(theory vs empirical) ·
    #   s13 §7a(NTK alignment: residual→NTK eigvec + NTK eigvec→FH right-singular-vecs)
    # Defaults match the widget's panel checkboxes: §1–§4, §6, §7, §7a, §9 default ON; the expensive
    # §5(SLQ) and §8, and the residual-projection sections §4b/§4c/§4d, default OFF.
    s1: int = 1
    s2: int = 1
    s3: int = 1
    s4: int = 1     # §4 J→H projections (on by default)
    s5: int = 0     # §5 SLQ spectra (expensive; off by default)
    s6: int = 0
    s7: int = 0     # §7 multi-sample NTK + FH tensor SVD (off by default)
    s8: int = 0     # §8 vec(J) onto FH-reshape singular vecs (off by default)
    s9: int = 0     # §4b  (multi-sample only; off by default)
    s10: int = 0    # §4c  (multi-sample only; off by default)
    s11: int = 0    # §4d  (multi-sample only; off by default)
    s12: int = 1    # §9 theory vs empirical sharpness (on by default)
    s13: int = 1    # §7a NTK alignment — residual→NTK eigvec + NTK eigvec→FH-SVD (on by default)
    s14: int = 1    # §9c σ₁ predictions vs the FULL loss-Hessian sharpness λmax(∇²L) (on by default)

    # ---- test set (held-out) ----
    n_test: int = 0                # held-out test points (0 ⇒ default: max(nsamp, 256), capped per dataset)

    # ---- logging / wandb ----
    wandb: int = 0                 # push metrics + panels to Weights & Biases
    wandb_project: str = "eos_lab"
    wandb_entity: str = ""         # blank ⇒ default entity
    run_name: str = ""             # blank ⇒ auto-built from hyperparameters

    def P(self):
        """Plain dict the numerics consume — string flags match server.py's _parse_params output."""
        d = asdict(self)
        d["bias"] = "1" if self.bias else "0"
        d["fixedx"] = "1" if self.fixedx else "0"
        d["gs"] = bool(self.gson)
        for k in ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14"):
            d[k] = bool(getattr(self, k))
        return d

    @classmethod
    def from_preset(cls, name, **overrides):
        if name not in PRESETS:
            raise KeyError(f"unknown preset '{name}'. Options: {', '.join(PRESETS)}")
        cfg = {**PRESETS[name], **overrides}
        return cls(**cfg)


# Mirrors index.html PRESETS. (The GPU presets default s5=0; turn on with s5=1 for SLQ.)
PRESETS = {
    "linear":    dict(dataset="synthetic", arch="mlp", act="linear", depth=4, width=100, bias=0,
                      fixedx=1, indim=10, outdim=1, nsamp=1, lr=0.3, init=0.1, steps=400,
                      s1=1, s2=1, s3=1, s5=0, s6=0),
    "msample":   dict(dataset="synthetic", arch="mlp", act="tanh", depth=4, width=100, bias=0,
                      fixedx=0, indim=10, outdim=1, nsamp=10, lr=0.3, init=0.1, steps=800,
                      s1=1, s2=1, s3=1, s5=0, s6=0),
    "gpu_run":   dict(dataset="synthetic", arch="mlp", act="tanh", depth=4, width=100, bias=0,
                      fixedx=0, indim=10, outdim=1, nsamp=25, lr=0.5, init=0.1, steps=2000,
                      s1=1, s2=1, s3=1, s5=0, s6=0),
    "cifar_mlp": dict(dataset="cifar10", arch="mlp", loss="mse", depth=2, width=128, act="tanh",
                      bias=0, fixedx=0, nsamp=128, lr=0.02, init=0.5, steps=300, eigevery=2,
                      slqprobes=3, s1=1, s2=1, s3=1, s5=0, s6=0),
    "cifar_cnn": dict(dataset="cifar10", arch="cnn", loss="mse", nsamp=128, lr=0.05, init=0.5,
                      steps=300, eigevery=2, slqprobes=3, chmul=1.0, s1=1, s2=1, s3=1, s5=0, s6=0),
    "cifar_vgg": dict(dataset="cifar10", arch="vgg11", loss="mse", nsamp=128, batch=128, lr=0.05, init=0.5,
                      steps=300, eigevery=2, slqprobes=3, chmul=0.25, gson=0, s1=1, s2=1, s3=1, s5=0, s6=0),
    "sort_gpt":  dict(dataset="sorting", arch="gpt", loss="mse", nsamp=128, batch=128, lr=0.0002, init=0.1,
                      steps=4000, eigevery=2, slqprobes=3, dmodel=64, nhead=4, nlayer=2, seqlen=16,
                      indim=16, outdim=16, s1=1, s2=1, s3=1, s5=0, s6=0),
    # OpenWebText next-token LM (GPT-2 BPE, cross-entropy). `nsamp` length-`seqlen` sequences form the
    # pool; each GD step samples a `batch` minibatch used for BOTH the step and that step's diagnostics.
    # CE disables the residual-geometry sections (§4/§6–§9); §1/§2/§3/§5 (loss/sharpness/eigs/SLQ) apply.
    "owt_gpt":   dict(dataset="owt", arch="gpt", loss="ce", nsamp=64, batch=16, lr=0.0006, init=0.1,
                      steps=2000, eigevery=5, slqprobes=3, dmodel=64, nhead=4, nlayer=2, seqlen=128,
                      vocab=50257, s1=1, s2=1, s3=1, s5=0, s6=0),
    # Chebyshev regression (Cohen et al. EoS): nsamp points on [-1,1] labeled by T_degree; the canonical
    # 1-hidden-layer tanh net (Linear(1,100)→tanh→Linear(100,1)). Sharpens to the 2/η edge of stability.
    "chebyshev": dict(dataset="chebyshev", arch="mlp", loss="mse", act="tanh", depth=1, width=100, bias=1,
                      fixedx=0, indim=1, outdim=1, nsamp=20, degree=3, lr=0.2, init=0.5, steps=500,
                      s1=1, s2=1, s3=1, s5=0, s6=0),
}
