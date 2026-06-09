# eos_lab — interpretable Python port of the EoS / Progressive-Sharpening widget

A **standalone, readable** Python package that does what the webpage (`index.html`) and its GPU
backend (`server.py`) do — train a small net full-batch with gradient descent and measure
progressive sharpening (PS) and the edge of stability (EoS) — but as a library you import and a CLI
you run on the GPU yourself. No browser, no HTTP server.

It is a **mirror**, not a fork: each module maps to a part of the webpage, and the numerics are
verified to match `server.py` (same `seed` ⇒ **identical GD trajectory**; see *Faithfulness* below).
So when you ask for a change here, the corresponding spot in `server.py` / `index.html` is obvious.

## Install / run

Uses the same env as `server.py` (`torch`, `numpy`, `matplotlib`). From the repo root:

```bash
PY=/nas/ucb/samsj/conda_env/envs/samsenv/bin/python

# CLI: run a preset on a GPU, save arrays + panels to runs/<preset>/
$PY -m eos_lab.cli --preset msample --device cuda:0
$PY -m eos_lab.cli --preset cifar_vgg --device cuda:0 --set chmul=0.5 --cifar-dir <cifar-dir>
$PY -m eos_lab.cli --list-presets

# library
$PY -c "from eos_lab import Config, run_job; r=run_job(Config.from_preset('msample'),device='cuda:0',progress=True)"

# many jobs across GPUs (one per array index, one GPU each)
sbatch --array=0-3 eos_lab/slurm/submit_array.sh
```

`run_job(cfg)` returns `{"meta", "config", "history", "series"}`:
`history` is the per-step records; `series` is tidy column arrays for plotting/saving.

## Module map — this package ↔ `server.py` ↔ `index.html`

| module            | what it is                                   | mirrors in `server.py`                          | webpage |
|-------------------|----------------------------------------------|-------------------------------------------------|---------|
| `rng.py`          | mulberry32 + Box–Muller gauss                | `u32…mulberry32`, `gauss`                        | `mulberry32`, `gauss` |
| `models.py`       | MLP/CNN/VGG11/GPT, losses, autograd grads/HVPs | `build_spec`,`fwd`,`*Model`,`MSE/CELoss`,`hvp*` | the architectures |
| `linalg.py`       | Lanczos, SLQ, principal angles, subspaces    | `_lanczos_core`,`lanczos_extreme*`,`slq_density`,`principal_angles` | §1–§3, §5, §6 |
| `data.py`         | synthetic / CIFAR-10 / sorting + seeded init | `load_cifar`,`load_sort`,`init_data_theta`      | the datasets |
| `config.py`       | `Config` dataclass + `PRESETS`               | `_parse_params` defaults                         | control panel + preset dropdown |
| `diagnostics.py`  | per-step section computations (§1–§9)        | the per-step block of `run_stream`               | all sections |
| `train.py`        | GD loop + `run_job` (+ test loss, `on_step`) | `run_stream` (loop + meta/step records)          | the **Run** button |
| `plots.py`        | matplotlib panels (all sections + train/test)| (browser draws these in Plotly)                  | the plot rows |
| `logging_wandb.py`| run naming + Weights & Biases logging        | (n/a — headless sweeps)                          | (n/a) |
| `cli.py`          | command-line runner (`--wandb`, auto-naming) | the HTTP `/run` handler                          | pressing Run |
| `slurm/gen_grid.py`| generate the SBATCH run grid + `submit_all` | (n/a)                                            | (n/a) |

## Sections — all ported (full widget parity)

Every widget toggle is implemented and **on by default**. Section ↔ flag map (see `config.Config`):

- **§1** `s1` loss (train **+ held-out test**) · **sharpness** (= λ_max of the loss Hessian) vs `2/η` · residual · spectral edges of `H`
- **§2 / §3** `s2`/`s3` top-/bottom-`n` eigenvalues of `H`, the loss Hessian, Gauss–Newton `G`, residual term `S`  (loss Hessian = `G + S`)
- **§4** `s4` `J = ∇Σf` projected onto top/bottom eigenvectors of `H`  (raw + Frobenius-normalized)
- **§5** `s5` SLQ spectral density of `H`, loss Hessian, `G`, `S`
- **§6** `s6` eigenspace rotation of `H` — principal angles of the ±-energy subspaces, step to step
- **§7** `s7` multi-sample NTK & function-Hessian tensor SVD
- **§8** `s8` `vec(J)` onto right-singular vectors of the `p×(p·dₙ)` FH reshape
- **§4b/§4c/§4d** `s9`/`s10`/`s11` `J·r` onto eigvecs of `Q[u₁]` · `Q[u₁]·(J·r)` onto GN eigvecs · per-residual-sign-group projections
- **§9** `s12` theoretical (Eq-13/21/27/29) vs empirical σ₁ over frozen-Q windows

The multi-sample sections (§7/§8/§4b–d and the multi branch of §9) form the explicit `M×p` Jacobian, so
they are **auto-skipped** when `M = N·d_out` or `M·p` is too large (gated by `Diagnostics.multi_ok`,
`M ≤ 512`, `M·p ≤ 2e8`); the skipped set is recorded in `meta["sections_skipped"]` and logged. The
matrix-free sections (§1–§6, §9-single) always run, so large-`N` runs still produce the full PS/EoS story.

## Logging, naming & sweeps

- **`run_name(cfg)`** (in `logging_wandb.py`) builds a deterministic, filesystem-safe name from the
  hyperparameters; the CLI uses it for both the local folder (`--out-root/<run_name>/`) and the wandb run.
- **`--wandb`** streams every per-step metric (all section vectors expanded to `base/i`) to project
  `eos_lab`, and logs every rendered panel image at the end. Degrades to local-only if wandb is unavailable.
- **`eos_lab/slurm/gen_grid.py`** writes one SBATCH script per run (datasets × architectures × `nsamp ∈
  {1,10,100,1000,10000}` × seeds) into `eos_lab/slurm/runs/`, plus `submit_all.sh`. Export `WANDB_API_KEY`
  (or put it in `~/.wandb_key`) first; `sbatch` propagates it.

## Faithfulness

For a synthetic MLP at a fixed `seed`, `eos_lab` reproduces `server.py` exactly, and all ported
projection/theory sections (§4–§9) match too:

```
max |Δloss|      = 0.00e+00   (GD trajectory bit-identical: same mulberry32 init/data + same gradient)
max |Δsharpness| ≈ 1e-8       (Lanczos on autograd HVPs vs server's finite-difference HVPs)
§4–§9 vectors    Δ = 0 (autograd archs) … ≤1e-6 (MLP, FD-vs-autograd)   [verified vs server.run_stream]
```

Two intentional differences from `server.py`, both for clarity:

1. **Explicit `device`/`dtype`** are threaded through every function (no thread-locals). More lines,
   but you can always see where a tensor lives.
2. **Autograd HVPs** (`torch.autograd` double-backward) instead of finite differences. Exact and
   short to read; the GD trajectory is unaffected (only the curvature *probes* differ, and autograd
   is the more accurate of the two).

## Notes

- Default dtype: fp64 on CPU, fp32 on GPU (matches `server.py`). Pass `--dtype float64` to match the
  browser’s double precision exactly.
- Parameter cap `p ≤ 10,000,000` (matrix-free, so memory is `O(p)`).
- CIFAR needs the raw pickle batches — pass `--cifar-dir` (same dir `server.py` uses).
