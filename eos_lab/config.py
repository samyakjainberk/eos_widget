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
    dataset: str = "synthetic"      # synthetic | cifar10 | cifar2 | mnist | mnist2 | sorting | owt | chebyshev | ksparse
    arch: str = "mlp"               # mlp | cnn | vgg11 | gpt
    loss: str = "mse"               # mse | ce
    chmul: float = 0.25             # CNN/VGG channel multiplier
    dmodel: int = 64                # GPT d_model
    nhead: int = 4                  # GPT heads
    nlayer: int = 2                 # GPT blocks
    seqlen: int = 16                # sorting sequence length / owt block size (tokens per sequence)
    vocab: int = 50257              # owt vocabulary (GPT-2 BPE) — token-embedding/head size
    degree: int = 3                 # chebyshev: degree of the target Chebyshev polynomial T_k
    ksparse: int = 3                # k-sparse parity: #bits in the parity subset S (≤ indim = #input bits)
    c2a: int = 0                    # cifar2: first CIFAR-10 class index → scalar label +1
    c2b: int = 1                    # cifar2: second CIFAR-10 class index → scalar label −1
    cvar: float = 0.0               # const dataset: variance of Gaussian noise added to the constant target (0 ⇒ exact constant)

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
    optimizer: str = "gd"           # gd (full-batch GD) | sign (θ←θ−lr·sign∇L) | spectral (Muon-style: per weight
                                    #   matrix step lr·UVᵀ from svd(∇W)=UΣVᵀ, biases→sign; NO momentum). Only the
                                    #   actual trajectory/statistics change — §9/§10 theory keeps the GD update.
    init: float = 0.1               # init magnitude; meaning depends on initscheme (below)
    initscheme: str = "default"     # default(=scale/√fan_in) | mup | xavier_normal | xavier_uniform |
                                    #   custom(=Gaussian std = init). `init` is the scale/gain/std per scheme.
    steps: int = 400
    seed: int = 0

    # ---- analysis (mirror 'Analysis & simulation' panel) ----
    neig: int = 3                   # top/bottom-n eigenpairs
    kth: int = 1
    eigevery: int = 1               # compute diagnostics every k GD steps
    heavyevery: int = 4             # cadence (in diagnostic ticks) for the heaviest slowly-varying multi-sample
                                    #   panels §7 (FH-eigvec projections) and §8 (FH-reshape SVD) — they are far
                                    #   costlier per step than the rest, so computing them every `heavyevery`-th
                                    #   tick (instead of every tick) keeps a run responsive while the cheap core
                                    #   (loss/sharpness/eigenvalues/§9 theory/§7a alignment) stays every tick.
                                    #   §5 SLQ uses its own coarser stride (≈50 snapshots/run). Set 1 to disable.
    slqprobes: int = 4
    energyp: float = 99.0           # §6 subspace energy %
    gson: int = 1                   # compute G & S spectra in §2/§3
    qapprox: int = 25              # §9 frozen-Q window length (steps)
    qmode: int = 1                 # §9 col-3 mode index — legacy (Eq-22 uses no mode)
    tset: int = 3                  # §9 Eq-29 number of top modes |T|
    grid3dcap: int = 500           # §11 3D-grid cap: skip the N×N×N Hessian–NTK grids when M=N·d_out exceeds this.
    sec12ncap: int = 500           # §12 sample cap: gate the per-sample Hessian eigvec cross-similarity on N (no p-cap;
                                   #   §12 builds dense p×p Hessians + O(p³) eig per sample, so large p is slow).
                                   #   Costs M Hessian-vector products; only the top-|value| points are emitted/plotted
                                   #   (G3D_MAXPTS) so even M≈100 (M³≈10⁶ entries) stays renderable.
    cubicapprox: int = 10          # §10 cubic-approximation window: every `cubicapprox` steps freeze θ₀, J₀, Q₀
                                   #   and the 3rd-derivative tensor T, then propagate BOTH J and Q over the window
                                   #   (Eqs 44/45 single, 49/50 multi) and predict the NTK-σ₁ change (Eq-47 single /
                                   #   Eq-51 multi). Smaller = more frequent re-anchoring (and cheaper: cost is
                                   #   O(window²) Jacobian-HVPs because Q is propagated exactly).

    # ---- which sections to compute ----
    # Section ↔ widget toggle map:  s1 §1 · s2 §2(top) · s3 §3(bottom) · s4 §4(J→H eigvecs) ·
    #   s5 §5(SLQ) · s6 §6(rotation) · s7 §7(multi-sample NTK/FH SVD) · s8 §8(FH-reshape SVD) ·
    #   s9 §4b(J·r→Q[u₁]) · s10 §4c(Q[u₁]·Jr→GN) · s11 §4d(per-sign-group) · s12 §9(theory vs empirical) ·
    #   s13 §7a(NTK alignment: residual→NTK eigvec + NTK eigvec→FH right-singular-vecs)
    # Defaults match the widget's panel checkboxes: §1–§4, §6, §7, §7a, §9 default ON; the expensive
    # §5(SLQ) and §8, and the residual-projection sections §4b/§4c/§4d, default OFF.
    # Every section is ON by default so a plain run renders every available panel. The expensive ones
    # (§5 SLQ, §7/§8 and §4b–d which build the M×p Jacobian) still skip themselves automatically when
    # infeasible (CE loss, single sample, or M·p over budget — see meta["sections_skipped"]); set any
    # flag to 0 to skip it explicitly and speed up a run / sweep.
    s1: int = 1
    s2: int = 1
    s3: int = 1
    s4: int = 1     # §4 J→H projections
    s5: int = 1     # §5 SLQ spectra (expensive)
    s6: int = 1     # §6 eigenspace rotation (principal angles)
    s7: int = 1     # §7 multi-sample NTK + FH tensor SVD (multi-sample only)
    s8: int = 1     # §8 vec(J) onto FH-reshape singular vecs (multi-sample only)
    s9: int = 1     # §4b  (multi-sample only)
    s10: int = 1    # §4c  (multi-sample only)
    s11: int = 1    # §4d  (multi-sample only)
    s12: int = 1    # §9 theory vs empirical sharpness
    s13: int = 1    # §7a NTK alignment — residual→NTK eigvec + NTK eigvec→FH-SVD (on by default)
    s14: int = 1    # §9c σ₁ predictions vs the FULL loss-Hessian sharpness λmax(∇²L) (on by default)
    s15: int = 1    # §9d σ₁ predictions with the residual self-computed by the quadratic model (on by default)
    s16: int = 1    # §9d-c §9d predictions vs the full loss-Hessian sharpness (on by default)
    s18: int = 0    # §11 3D GRIDS (multi-sample): three N×N×N tensors over sample indices (i,j,k) —
                    #   T1=Jᵢᵀ Qⱼ Jₖ, T2=uⱼuₖ·T1, T3=rᵢuⱼuₖ·T1 (u=top NTK eigvec, r=residual). OFF by default
                    #   (M³ points + M HVPs — small-M only; gated by grid3dcap), computed on the SLQ cadence.
    s17: int = 0    # §10 CUBIC approximation: Eq-47 (single) / Eq-51 (multi) σ₁ predictions ±PSD, Eq-51 without
                    #   the η² term, and ‖ΔQ‖/‖ΔJ‖ over the window — vs both the NTK σ₁ (panel 1) and the full
                    #   loss-Hessian λmax (panel 2). OFF by default: it propagates Q exactly (O(window²) HVPs/window)
                    #   so it is by far the heaviest section; enable per-run when you want the cubic comparison.
    s19: int = 0    # §12 per-sample Hessian Q_i=∇²f_i eigenvector cross-similarity grids/cuboids + principal
                    #   angles over sample pairs (i,j). OFF by default; forms N dense p×p Hessians (p HVPs) + N
                    #   eigendecompositions — small N (≤grid3dcap) AND small p (≤sec12pcap) only.
    s20: int = 0    # §14 Tr(ΔNTK) per-triplet (i,j,k) decomposition cubes — reuses §12's per-sample Lanczos
                    #   eigenpairs; small N (≤grid3dcap). OFF by default.
    s21: int = 0    # §13 residual-weighted curvature G1/G2/G3 vs the exact reference J_iᵀQ_jJ_k·r_k — reuses §12's
                    #   per-sample Lanczos eigenpairs but adds N HVPs for the exact reference; small N (≤grid3dcap).
                    #   OWN toggle (independent of §12/s19); triggers the shared Lanczos by itself. OFF by default.
    s22: int = 0    # §12b — per-sample projection panels (product |⟨J,u⟩|·σ·r & alignment |⟨J,u⟩|/‖J‖). s19 is now
                    #   §12a (principal angles + diagonal alignment); s22 is §12b. Both share the §12 Lanczos. OFF by default.
    s23: int = 0    # §15 — 2nd-difference decomposition of ‖J‖²_F & σ₁ into theory terms I/II/III (resp. IV/V/VI) +
                    #   matrices A/B. Holds 3 consecutive eig-ticks; multi-sample only; small-N (≤grid3dcap). OFF by default.
    divreg: float = 0.3   # §15 divergence ε: softens the 0/0 spike where D²→0 at ‖J‖²/σ₁ inflections (0 ⇒ exact /D²).
    s24: int = 0    # §16 — standalone curvature-aligned per-residual-sign optimizer (own iteration axis). MLP+MSE+scalar,
                    #   small N & p. OFF by default.
    s24warm: int = 5      # §16: standard-GD warmup steps from θ₀ before the §16 optimization begins.
    s24iter: int = 250    # §16: number of §16 iterations after the warmup.
    s24grid: float = 0.1  # §16: (β,s) search grid step (finer ⇒ more descent, costs (1/step)² loss-evals).
    s24ares: float = 0.01 # §16: α line-search resolution (finer ⇒ smaller beneficial steps near the plateau).
    s24k: int = 8         # §16/§17: # eigvecs per side (top-kdir & bottom-kdir) onto which g₊/g₋ are projected (coeff ⟨g,u_i⟩, NO σ/η).
    s24base: int = 0      # §16: compute the five dotted baselines (A=random,B=shuffle,C=randfix,D=eigfix,E=gd). OFF (≈6× cost).
    s25: int = 0          # §17 — PER-SAMPLE function-Hessian variant of §16 (each sample uses its OWN Q_k eigvec). O(M²); small M. OFF.
    s25base: int = 0      # §17: compute the five dotted baselines (per-sample analog of §16's). OFF (≈6× cost).

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


# Mirrors index.html PRESETS. Section flags are inherited from the all-on Config defaults
# (every panel renders); set e.g. s5=0 here or via --set to skip SLQ on a heavy preset.
PRESETS = {
    "linear":    dict(dataset="synthetic", arch="mlp", act="linear", depth=4, width=100, bias=0,
                      fixedx=1, indim=10, outdim=1, nsamp=1, lr=0.3, init=0.1, steps=400),
    "msample":   dict(dataset="synthetic", arch="mlp", act="tanh", depth=4, width=100, bias=0,
                      fixedx=0, indim=10, outdim=1, nsamp=10, lr=0.3, init=0.1, steps=800),
    "gpu_run":   dict(dataset="synthetic", arch="mlp", act="tanh", depth=4, width=100, bias=0,
                      fixedx=0, indim=10, outdim=1, nsamp=25, lr=0.5, init=0.1, steps=2000),
    "cifar_mlp": dict(dataset="cifar10", arch="mlp", loss="mse", depth=2, width=128, act="tanh",
                      bias=0, fixedx=0, nsamp=128, lr=0.02, init=0.5, steps=300, eigevery=2,
                      slqprobes=3),
    # 2-class SCALAR CIFAR (classes c2a,c2b → ±1, d_out=1) — small M=N so EVERY section incl the per-sample §16/§17 runs
    "cifar2_mlp": dict(dataset="cifar2", arch="mlp", loss="mse", act="tanh", bias=1, fixedx=0, depth=1,
                       width=64, nsamp=16, lr=0.02, init=0.5, steps=300, eigevery=2, slqprobes=3,
                       s18=1, s19=1, s20=1, s21=1, s22=1, s23=1, s24=1, s25=1),
    # MNIST 10-class one-hot (padded to 3×32×32 → drop-in for cifar-shaped nets). Small N + small width so the
    # per-sample §11–§16 sections fit (their batched Lanczos costs ~M·mV·p, and M=N·10 inflates it). §17 needs
    # M=N·d_out ≤ 64 (so N ≤ 6) — use the scalar mnist2 preset for §17.
    "mnist_mlp": dict(dataset="mnist", arch="mlp", loss="mse", act="tanh", bias=1, fixedx=0, depth=1,
                      width=48, nsamp=8, lr=0.02, init=0.5, steps=300, eigevery=2, slqprobes=3,
                      s18=1, s19=1, s20=1, s21=1, s22=1, s23=1, s24=1),
    # 2-class SCALAR MNIST (classes c2a,c2b → ±1, d_out=1) — small M=N so every section incl §16/§17 runs
    "mnist2_mlp": dict(dataset="mnist2", arch="mlp", loss="mse", act="tanh", bias=1, fixedx=0, depth=1,
                       width=64, nsamp=16, lr=0.02, init=0.5, steps=300, eigevery=2, slqprobes=3,
                       s18=1, s19=1, s20=1, s21=1, s22=1, s23=1, s24=1, s25=1),
    "cifar_cnn": dict(dataset="cifar10", arch="cnn", loss="mse", act="tanh", nsamp=128, lr=0.05, init=0.5,
                      steps=300, eigevery=2, slqprobes=3, chmul=1.0),
    "cifar_vgg": dict(dataset="cifar10", arch="vgg11", loss="mse", act="tanh", nsamp=128, batch=128, lr=0.05, init=0.5,
                      steps=300, eigevery=2, slqprobes=3, chmul=0.25, gson=0),
    "sort_gpt":  dict(dataset="sorting", arch="gpt", loss="mse", nsamp=128, batch=128, lr=0.0002, init=0.1,
                      steps=4000, eigevery=2, slqprobes=3, dmodel=64, nhead=4, nlayer=2, seqlen=16,
                      indim=16, outdim=16),
    # OpenWebText next-token LM (GPT-2 BPE, cross-entropy). `nsamp` length-`seqlen` sequences form the
    # pool; each GD step samples a `batch` minibatch used for BOTH the step and that step's diagnostics.
    # CE disables the residual-geometry sections (§4/§6–§9); §1/§2/§3/§5 (loss/sharpness/eigs/SLQ) apply.
    "owt_gpt":   dict(dataset="owt", arch="gpt", loss="ce", nsamp=64, batch=16, lr=0.0006, init=0.1,
                      steps=2000, eigevery=5, slqprobes=3, dmodel=64, nhead=4, nlayer=2, seqlen=128,
                      vocab=50257),
    # Chebyshev regression (Cohen et al. EoS): nsamp points on [-1,1] labeled by T_degree; a 6-hidden-layer
    # width-50 tanh MLP (≈12.9k params). Sharpens to the 2/η edge of stability.
    "chebyshev": dict(dataset="chebyshev", arch="mlp", loss="mse", act="tanh", depth=6, width=50, bias=1,
                      fixedx=0, indim=1, outdim=1, nsamp=20, degree=3, lr=0.1, init=0.5, steps=5000),
}
