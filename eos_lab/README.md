# eos_lab — the Progressive-Sharpening / EoS lab as a Python package

`eos_lab` does everything the browser widget (`index.html`) and its GPU backend (`server.py`) do —
train a small network with full-batch gradient descent and watch **progressive sharpening (PS)** and the
**edge of stability (EoS)** unfold — but as a plain Python library and command-line tool. No browser, no
HTTP server: import it, or point it at a GPU and get back arrays and figures.

It mirrors the widget faithfully: at the same `seed` the gradient-descent trajectory is **bit-identical**
and every diagnostic is checked against `server.py`, so a run here and a
run in the browser are directly comparable.

## Running it

Needs the same dependencies as `server.py`, plus `matplotlib` for the figures:

```bash
pip install torch numpy matplotlib
```

Run these from the **repo root** (the folder that contains the `eos_lab/` directory):

```bash
# Run a preset on a GPU; arrays + panel images land in runs/<preset>/
python3 -m eos_lab.cli --preset msample --device cuda:0
python3 -m eos_lab.cli --preset cifar_vgg --device cuda:0 --set chmul=0.5 --cifar-dir <cifar-dir>
python3 -m eos_lab.cli --preset mnist2_mlp --device cuda:0     # 2-class scalar MNIST — runs the per-sample §16/§17
python3 -m eos_lab.cli --preset cifar2_mlp --device cuda:0 --set c2a=3 --set c2b=5   # pick which two classes → ±1
python3 -m eos_lab.cli --list-presets

# As a library
python3 -c "from eos_lab import Config, run_job; r = run_job(Config.from_preset('msample'), device='cuda:0', progress=True)"

# A sweep across GPUs (one job per array index)
sbatch --array=0-3 eos_lab/slurm/submit_array.sh
```

`run_job(cfg)` hands back `{"meta", "config", "history", "series"}` — `history` is the per-step records,
and `series` is the same numbers as tidy column arrays, ready to plot or save.

## What's inside

A handful of small, single-purpose modules:

- **`config.py`** — the `Config` dataclass and the named presets.
- **`models.py`** — the networks (MLP, CNN, VGG11, mini-GPT), the MSE / cross-entropy losses, and exact gradients + Hessian-vector products via autograd.
- **`data.py`** — the datasets (synthetic, CIFAR-10, **CIFAR-2** = 2-class scalar ±1, **MNIST** & **MNIST-2** [padded to 3×32×32], sorting, OpenWebText, Chebyshev, **Chebyshev-2** [`chebyshev2`: same Tₖ target via the closed form cos(k·arccos x)], **k-sparse parity** [`ksparse`: `in_dim` ±1 bits → scalar ±1 = parity of a fixed random `k`-subset; works with the MLP and the mean-pooled mini-GPT]) and seeded initialization.
- **`linalg.py`** — the matrix-free linear algebra: Lanczos, stochastic Lanczos quadrature, principal angles, subspaces, and a robust `safe_eigh`/`safe_eigvalsh` (ramp + numpy fallback) so degenerate scalar-output Hessians never crash LAPACK/cuSOLVER.
- **`diagnostics.py`** — the per-step measurements behind every panel (§1–§15, plus the ported §18–§26: `sec18`–`sec21`, `sec24`, `sec25`, `sec26`).
- **`sec16.py` / `sec17.py`** — the two standalone curvature-aligned optimizers (§16 averaged-Hessian, §17 per-sample): project `g₊`/`g₋` onto the top-/bottom-`kdir` eigvectors (coefficient `⟨g,uᵢ⟩`, no `σ/η`), pure-curvature main run, with their five baselines (incl. E·gd).
- **`prediction.py`** — the **`/prediction`-widget forecasts** ported from `server.py`, driven by **`run_predictions(cfg, device, dtype)`** (one optimizer trajectory from θ₀ + every enabled tracker stepped per tick; gated by the `s36`/`s37`/`s38`/`s39` config toggles). Includes the `(lr, init-std)` **`sweep()`** (predictions 1 / 2 / 2b — the cubic & quadratic sign-change forecasts `tPred`/`tPredQ`, the `swsumn`-window sums `sumD`/`sumCurv`/`sumD2`/`sumJgrad`, `alignNTK`), **`pred_signchange(…, cubic=…, opt=…)`** (the `‖J·ṙ‖−‖J̇·r‖` sign-change forecast), and four per-step trackers: **`Pred3Tracker`** (frozen-`J*` residual theory — 1st + 2nd order + the **3rd evolving-`J` step-function** theory + `p3rep` re-anchor + the `t*∓10/+20` freeze sweep), **`Pred4Tracker`** (frozen-`M_r` vs the **evolving-Qr** self-model NTK-eigenvalue propagation + NTK/Gauss-Newton eigenvector cosines + `p4rep` + freeze sweep), **`TraceTracker`** (`Tr(NTK)` quad/cubic × live/self vs `Tr(∇²L)`/`Tr(JᵀJ)`), and the new **`Pred6Tracker`** (4 adaptive-optimizer trajectories {GD, Signed-GD, Muon, Gauss-Newton} from a shared θ₀ — top-50 Gauss-Newton scree + loss + Lanczos sharpness per trajectory). Every tracker + the sweep threads the run's `optimizer` through the preconditioner library in `models.py` — **`matpow_sym` / `muon_whiten` / `gn_precond` / `opt_dir` (4-way incl. `gaussnewton`) / `precond_flow`**, all **bit-identical** to the server (`gd ⇒ (Jᵀr, η/N)` byte-identical). **Parity:** for **gd** and **gaussnewton**, the exact / HVP-free / eigenvalue pieces match the server to ~1e-9…1e-16 (Pred-3 theories ~1e-16, sweep integer forecasts **exact**, Pred-6 scree ~1e-11) and the `hvp_S`/`hvp_L`-routed quantities (Pred-4 `M_r·J`, Trace `trHess`) carry the documented **~1e-3** FD-vs-exact gap. **Signed-GD, Muon, and any preconditioner-in-the-loop *predicted* trajectory (Pred-4's `kPred` under GN) are not bit-reproducible** — `sign(0)` flips on exactly-zero gradient components and Muon/GN `eigh` amplify the inherent ~1e-16 ∇L non-associativity along the chaotic EoS trajectory (the server's own fp32↔fp64 shows the same); use **absolute** diff for the ∈[0,1] eigenvector cosines.
- **`train.py`** — the gradient-descent loop and `run_job`.
- **`plots.py`** — renders each panel as a matplotlib figure.
- **`cli.py`** / **`logging_wandb.py`** — the command-line runner and optional Weights & Biases logging.
- **`slurm/`** — scripts to fan a sweep out across GPUs.

Nothing here ever forms a `p×p` matrix: every curvature measurement is a Hessian-vector product, so memory
stays `O(p)` and runs scale to multi-million-parameter networks.

## What each panel shows

`eos_lab` **computes §1–§26** — a full mirror of `server.py`. §1–§17 are described below; the later
sections **§18–§26** (`s26`–`s34`) were ported and are computed per-step by `diagnostics.py` (`g18`–`g21`,
`g24`, `g25`, `g26`), serving as the byte-parity reference for those panels. Exact-autograd/Jacobian
quantities match `server.py` to ~1e-13; the finite-difference-HVP fields (§25 `jdr`/`ddJr`, §26 `M_r`
eigvecs) carry the documented FD-vs-exact gap because server's MLP uses finite-difference HVPs (for browser
byte-parity) while eos_lab uses exact autograd. §22/§23 (`s30`/`s31`) drive the quadratic-surrogate REPLACE
mode in `train.py`. Together with §10 (`s17`, cubic) and the §16/§17 baselines (`s24base` / `s25base`) the
opt-in sections default **OFF**. The §1–§17 flags below are each a flag on `config.Config`, all on by default:

- **§1** — loss (train **and** held-out test), **sharpness** (the top eigenvalue of the loss Hessian) against the `2/η` threshold, the residual, and the spectral edges of the function Hessian `H`.
- **§2 / §3** — the top / bottom `n` eigenvalues of `H`, the loss Hessian, the Gauss–Newton matrix `G`, and the residual term `S` (the loss Hessian is `G + S`).
- **§4** — the Jacobian `J = ∇Σf` projected onto the top/bottom eigenvectors `u` of `H`, in **3 phases**: phase 1 alignment `|⟨J,u⟩|/‖J‖`, phase 2 power-iteration `|⟨J,u⟩|·σ`, phase 3 residual-dominance `|⟨J·r,u⟩|·σ` — with `σ÷N` (per-sample scale) and the single-sample residual kept signed.
- **§5** — the spectral density (via stochastic Lanczos quadrature) of `H`, the loss Hessian, `G`, and `S`.
- **§6** — how fast the dominant eigenspace of `H` rotates from step to step (principal angles).
- **§7a** — NTK alignment: the residual projected onto the top-`n` NTK eigenvectors `vₖ` (`ntkR`) and the top NTK eigenvector onto the FH right-singular vectors (`ntkH`), plus two change-product diagnostics — the per-step product of the sharpness change and the projection change `Δλ(t)·Δ⟨r,vₖ⟩(t)` (`ntkGs`, with running average `ntkGsA`) and its one-step-lagged form `Δλ(t)·Δ⟨r,vₖ⟩(t−1)` (`ntkGl`/`ntkGlA`).
- **§7 / §8** — the multi-sample NTK and function-Hessian SVD, and `vec(J)` onto the FH-reshape singular vectors.
- **§4b / §4c / §4d** — residual-weighted Jacobian projections (multi-sample only).
- **§9** — the theory: each σ₁ prediction (Eqs. 13/21/22/23/29) overlaid on the measured top NTK / Gauss–Newton eigenvalue, over frozen-`Q` windows. The records also carry **§9b** (`thPpsd`: the predictions plus the second-order PSD term `‖ΔJᵀu₁‖²`, an always-positive sharpening floor the first-order recursion drops).
- **§9c** — the same predictions compared against the **full loss-Hessian sharpness** `λmax(∇²L)` (`thAH`) rather than the Gauss–Newton edge; the gap between them is the residual term `S`.
- **§9d / §9d-c** — the *same* Eq. 13/21/22/23/29 predictions (`thP_d`/`thPpsd_d`), but the residual driving the recursion is **self-computed by the frozen quadratic model** rather than read from the live run: inside each window `r_q = Y − f_quad(θ₀+Δθ)` with `f_quad = f₀ + J₀Δθ + ½ΔθᵀQΔθ` and `Δθ` advanced by the quadratic model's own gradient descent — a fully closed prediction. §9d overlays them on the measured NTK σ₁ (like §9); §9d-c on the full-Hessian sharpness (like §9c). They coincide with §9 at each window start (`Δθ=0`) and diverge by exactly the quadratic-approximation error.
- **§10 (cubic approximation, `s17`, OFF by default)** — the **cubic** model (paper §6). Every `cubicapprox` steps it freezes `θ₀`, `J₀`, `Q₀` and the third-derivative tensor `T` (matrix-free, central differences of the HVPs) and propagates **both** `J` and `Q` exactly, predicting the NTK-σ₁ change: **Eq. 47** (single, `c47`/`c47p`) and **Eq. 51** (multi, `c51`/`c51p`) ±the PSD term, plus Eq. 51 *without* the η² cubic term (`c51n`/`c51np` = the quadratic recursion). Two panels overlay them on the measured NTK σ₁ (`cActN`, + `‖ΔQ‖`/`‖Q₀‖` drift `cdQ`) and on the full-Hessian λmax (`cActH`, + `‖ΔJ‖`/`‖J₀‖` drift `cdJ`). It is **off by default** because propagating `Q` exactly costs `O(window²)` HVPs/window (by far the heaviest section); enable with `--set s17=1`.
- **§11 (`s18`)** — 3D Hessian–NTK grids `T1=JᵢᵀQⱼJₖ`, `T2=uⱼuₖT1`, `T3=rᵢuⱼuₖT1`, with per-`(i,j,k)`-class evolution stats (and rotating-GIF cuboids in `plots.py`). Multi-sample, small `N` (N³ grid, capped by `grid3dcap`).
- **§12a / §12b (`s19` / `s22`)** — the per-sample function Hessian `Qᵢ=∇²fᵢ` (one matrix-free Lanczos per sample): top/bottom-eigvec **principal angles** + diagonal alignment across samples (12a), and per-sample **projection** panels of `∇f` / `J·r` onto each `Qᵢ`'s eigvecs (12b). Both share one §12 Lanczos; small `N` (capped by `sec12ncap`).
- **§13 (`s21`)** — residual-weighted curvature `G1/G2/G3 ≈ ∑ JᵢᵀQⱼJₖ·rₖ` vs the exact reference. **§14 (`s20`)** — `Tr(ΔNTK)` per-triplet decomposition over the `N³` cube. Both reuse §12's per-sample eigenpairs.
- **§15 (`s23`)** — 2nd-difference decomposition of `‖J‖²_F` (=tr NTK) and `σ₁` into theory terms I/II/III (resp. IV/V/VI), the matrices A/B, the chained-contraction norms, and a per-**effective**-sample top/bottom projection ratio. MSE, single or multi.
- **§16 (`s24`, OFF by default)** — a standalone **curvature-aligned per-residual-sign optimizer** started from θ₀ (`sec16.py` / `plot_section16`): a GD warmup, then **project** the positive-residual gradient `g₊` onto the **top-`kdir`** eigvectors of the sample-averaged function Hessian `H̄` and `g₋` onto the **bottom-`kdir`** (coefficient `⟨g,uᵢ⟩`, *no* `σ/η` weighting), then `α₊/α₋` line-searches + a `(β,s)` grid. The **main run is pure curvature** (no gradient info). 6 panels (loss · loss-Hessian eig · function-Hessian eig · per-sample residuals · held-out test loss · ‖update‖₂) × {pos, neg, mean, best} look-aheads. `--set s24base=1` adds 5 dotted **baselines**: A random dirs · B shuffled ± sets · C frozen-random · D frozen-`H̄`-eigvec · **E the plain GD step**. `--set s24k=K` sets the eigvectors per side (default 32; the curvature extremes hold only a fraction of the gradient, so more directions descend better — the Lanczos subspace scales as `4·K+32` so they're actually resolved). MSE, small `M = N·d_out` (≤ 256).
- **§17 (`s25`, OFF by default)** — the **per-sample** variant of §16 (`sec17.py` / `plot_section17`): each sample projects `rₖ∇fₖ` onto the top-/bottom-`kdir` eigvectors of its **own** `Qₖ=∇²fₖ` (`r>0` → top, `r<0` → bottom) instead of the averaged `H̄`. Same 6 panels + the 5 baselines (`--set s25base=1`, incl. E·gd). Per-sample Lanczos every iteration, so `M = N·d_out` ≤ 256 (and ≤ 64 in the single-threaded browser).

The following **§18–§26** (`s26`–`s34`, all OFF by default) were ported from `server.py`; eos_lab computes them per-step in `diagnostics.py` (small `N`, MSE where noted). They need the `M×p` Jacobian + matrix-free HVPs:

- **§18 (`s26`)** — per-sample (sample1 vs sample2) projections of the §12 cube; emitted as `g18` (needs exactly `nsamp=2`).
- **§19 (`s27`, GD+MSE)** — one-step grad-norm change `A_t=Δ‖∇L‖` (the MSE-GD prediction) and `tr(NTK)`; `g19`.
- **§20 (`s28`)** — spectral histograms of the residual-weighted function Hessian `M_r=Σ_k r_kQ_k` (its eigenvalues `λ_i` and `λ_i·|⟨v_i,u_k⟩|` for the top Jacobian directions `u_k`), top-`s28k`⊕bottom-`s28k` via matrix-free Lanczos on `hvp_S`; `g20`.
- **§21 (`s29`)** — residual↔spectrum alignment: residual onto the top NTK (`JJᵀ`) eigvectors, and `J·r` onto the `M_r` eigvectors; `g21`.
- **§22 / §23 (`s30` / `s31`)** — §20 on a **quadratic surrogate**: frozen-`Q` at iteration `s30t` (§22) or a random-Hessian rank-`s31r` surrogate (§23). These are REPLACE-mode drivers in `train.py` (they run their own trajectory).
- **§24 (`s32`)** — 1st/2nd-order Δf alignment: `A=JJᵀr` (1st) and `B=(η/2N)[(J·r)ᵀQ_k(J·r)]_k` (2nd) vs the residual and the top-4 NTK eigvectors; `g24`.
- **§25 (`s33`)** — gradient-norm & `d/dt(J·r)` evolution: an 8-plot, 2-panel layout — the product-rule split `‖M_r∇L‖`/`‖JᵀJ∇L‖` + `‖∇L‖` + §15 II/III, the pairwise cosines of the θ-gradients of 7 scalar quantities (incl. `∇‖ġ‖²`, `ġ=−H∇L`), their norms, `‖d/dt(J·r)‖₁`, and the residual-direction drift `cos(r_t, r_{t−k})` for `k∈{1,10,30,50,100}` plus a bold `cos(r_t, r_0)` reference; `g25` (MLP-only cosine panels via exact autograd).
- **§26 (`s34`)** — eigenvector-direction drift `|cos(v_i(t), v_i(t−k))|` (`k∈{10,20,30,50,100}`) for the top-3 eigenvectors of GN=`JᵀJ` and NTK=`JJᵀ` and the top-3⊕bottom-3 of `M_r`, with **eigenvector continuation** (each mode matched to the previous step so eigenvalue crossings don't cause spurious jumps) and an `M_r` small-eigenvalue gate; 4 panels × 3 plots × 5 lags; `g26`.

The multi-sample sections build an explicit `M×p` Jacobian (`M = N·d_out`), so they're **skipped
automatically** when that's too big to form (the budget is `M ≤ 2048` and `M·p ≤ 7e8`; whatever is skipped
is recorded in `meta["sections_skipped"]`). The matrix-free sections — §1–§6 and single-sample §9 — always
run, so even large-`N` runs still tell the full PS/EoS story. The **per-sample** sections (§12–§17) run a
batched Lanczos over the `M = N·d_out` effective samples, so their cost scales with `M`; on **multi-class**
data (`d_out = 10`) that grows fast, so the `cifar2` / `mnist2` (2-class **scalar**, `d_out = 1`) datasets and
the small-`N` presets (`cifar2_mlp`, `mnist_mlp`, `mnist2_mlp`) keep them feasible — §16 and §17 both need `M ≤ 256`
(§17 ≤ 64 in the single-threaded browser).

To keep runs fast the heaviest slowly-varying panels are throttled (matching the widget): §5 SLQ runs ~50×
per run and §7's FH-eigenvector projections + §8 run every `heavyevery` ticks (default 4); the cheap core
(loss/sharpness/eigenvalues/§9 theory/§7a) runs every tick. Set `--set heavyevery=1` to compute every panel
every step.

## Sweeps & logging

- `--wandb` streams every per-step metric to a Weights & Biases project and uploads the panel images at the end (it falls back to local-only if wandb isn't available). Run folders and run names are derived deterministically from the hyperparameters.
- `eos_lab/slurm/gen_grid.py` writes one SBATCH script per run (datasets × architectures × `nsamp` × seeds), plus a `submit_all.sh`.

## Faithfulness

At a fixed `seed`, `eos_lab` reproduces `server.py` to the bit on the GD trajectory, and every ported
diagnostic matches:

```
max |Δloss|              = 0            (identical mulberry32 init + data + gradient)
§1/§2/§3 sharpness & H/H_L/G/S eigvals   ~1e-2  (≈4%)   (exact autograd HVP vs the server's
                                                         finite-difference HVP — see below)
§4/§7/§8/§15/§16/§17 multi-sample        Δ = 1e-5 … 1e-13
```

There are two deliberate differences, both for readability: `device`/`dtype` are passed explicitly
everywhere (no thread-locals, so you can always see where a tensor lives), and Hessian-vector products use
autograd double-backward instead of finite differences (exact, and it doesn't touch the GD trajectory —
only the curvature *probes* differ).

That second difference is why the **§1/§2/§3 sharpness and `H`/`H_L`/`G`/`S` eigenvalues** differ from the
server backend by **~1e-2 (≈4%)**: `eos_lab` uses **exact autograd HVPs**, whereas `server.py`'s `MlpModel`
uses **finite-difference HVPs** (kept for byte-parity with the browser). The multi-sample primitives that
build an explicit Jacobian (§4 / §7 / §8 / §15 / §16 / §17) don't go through that probe and **match to
1e-5 … 1e-13**.

## Good to know

- Default precision is fp64 on CPU, fp32 on GPU (matching `server.py`); pass `--dtype float64` to match the browser exactly.
- Parameter cap: `p ≤ 10,000,000` (memory is `O(p)`).
- CIFAR needs the raw pickle batches — point `--cifar-dir` at the same directory `server.py` uses.
