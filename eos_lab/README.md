# eos_lab — the Progressive-Sharpening / EoS lab as a Python package

`eos_lab` does everything the browser widget (`index.html`) and its GPU backend (`server.py`) do —
train a small network with full-batch gradient descent and watch **progressive sharpening (PS)** and the
**edge of stability (EoS)** unfold — but as a plain Python library and command-line tool. No browser, no
HTTP server: import it, or point it at a GPU and get back arrays and figures.

It mirrors the widget faithfully: at the same `seed` the gradient-descent trajectory is **bit-identical**
and every diagnostic is checked against `server.py`, so a run here and a
run in the browser are directly comparable.

## Running it

Uses the same environment as `server.py` (`torch`, `numpy`, `matplotlib`).

```bash
PY=/nas/ucb/samsj/conda_env/envs/samsenv/bin/python

# Run a preset on a GPU; arrays + panel images land in runs/<preset>/
$PY -m eos_lab.cli --preset msample --device cuda:0
$PY -m eos_lab.cli --preset cifar_vgg --device cuda:0 --set chmul=0.5 --cifar-dir <cifar-dir>
$PY -m eos_lab.cli --list-presets

# As a library
$PY -c "from eos_lab import Config, run_job; r = run_job(Config.from_preset('msample'), device='cuda:0', progress=True)"

# A sweep across GPUs (one job per array index)
sbatch --array=0-3 eos_lab/slurm/submit_array.sh
```

`run_job(cfg)` hands back `{"meta", "config", "history", "series"}` — `history` is the per-step records,
and `series` is the same numbers as tidy column arrays, ready to plot or save.

## What's inside

A handful of small, single-purpose modules:

- **`config.py`** — the `Config` dataclass and the named presets.
- **`models.py`** — the networks (MLP, CNN, VGG11, mini-GPT), the MSE / cross-entropy losses, and exact gradients + Hessian-vector products via autograd.
- **`data.py`** — the datasets (synthetic, CIFAR-10, sorting, OpenWebText, Chebyshev) and seeded initialization.
- **`linalg.py`** — the matrix-free linear algebra: Lanczos, stochastic Lanczos quadrature, principal angles, subspaces.
- **`diagnostics.py`** — the per-step measurements behind every panel (§1–§9).
- **`train.py`** — the gradient-descent loop and `run_job`.
- **`plots.py`** — renders each panel as a matplotlib figure.
- **`cli.py`** / **`logging_wandb.py`** — the command-line runner and optional Weights & Biases logging.
- **`slurm/`** — scripts to fan a sweep out across GPUs.

Nothing here ever forms a `p×p` matrix: every curvature measurement is a Hessian-vector product, so memory
stays `O(p)` and runs scale to multi-million-parameter networks.

## What each panel shows

Every section from the widget is here, and each is a flag on `config.Config` (all on by default):

- **§1** — loss (train **and** held-out test), **sharpness** (the top eigenvalue of the loss Hessian) against the `2/η` threshold, the residual, and the spectral edges of the function Hessian `H`.
- **§2 / §3** — the top / bottom `n` eigenvalues of `H`, the loss Hessian, the Gauss–Newton matrix `G`, and the residual term `S` (the loss Hessian is `G + S`).
- **§4** — the Jacobian `J = ∇Σf` projected onto the top/bottom eigenvectors of `H` (raw and normalized).
- **§5** — the spectral density (via stochastic Lanczos quadrature) of `H`, the loss Hessian, `G`, and `S`.
- **§6** — how fast the dominant eigenspace of `H` rotates from step to step (principal angles).
- **§7 / §8** — the multi-sample NTK and function-Hessian SVD, and `vec(J)` onto the FH-reshape singular vectors.
- **§4b / §4c / §4d** — residual-weighted Jacobian projections (multi-sample only).
- **§9** — the theory: each σ₁ prediction (Eqs. 13/21/22/23/29) overlaid on the measured top NTK / Gauss–Newton eigenvalue, over frozen-`Q` windows. The records also carry **§9b** (`thPpsd`: the predictions plus the second-order PSD term `‖ΔJᵀu₁‖²`, an always-positive sharpening floor the first-order recursion drops).
- **§9c** — the same predictions compared against the **full loss-Hessian sharpness** `λmax(∇²L)` (`thAH`) rather than the Gauss–Newton edge; the gap between them is the residual term `S`.

The multi-sample sections build an explicit `M×p` Jacobian (`M = N·d_out`), so they're **skipped
automatically** when that's too big to form (the budget is `M ≤ 2048` and `M·p ≤ 7e8`; whatever is skipped
is recorded in `meta["sections_skipped"]`). The matrix-free sections — §1–§6 and single-sample §9 — always
run, so even large-`N` runs still tell the full PS/EoS story.

## Sweeps & logging

- `--wandb` streams every per-step metric to a Weights & Biases project and uploads the panel images at the end (it falls back to local-only if wandb isn't available). Run folders and run names are derived deterministically from the hyperparameters.
- `eos_lab/slurm/gen_grid.py` writes one SBATCH script per run (datasets × architectures × `nsamp` × seeds), plus a `submit_all.sh`.

## Faithfulness

At a fixed `seed`, `eos_lab` reproduces `server.py` to the bit on the GD trajectory, and every ported
diagnostic matches:

```
max |Δloss|       = 0          (identical mulberry32 init + data + gradient)
max |Δsharpness|  ≈ 1e-8       (autograd vs finite-difference curvature probes)
§4–§9 vectors     Δ = 0 … ≤1e-6
```

There are two deliberate differences, both for readability: `device`/`dtype` are passed explicitly
everywhere (no thread-locals, so you can always see where a tensor lives), and Hessian-vector products use
autograd double-backward instead of finite differences (exact, and it doesn't touch the GD trajectory —
only the curvature *probes* differ).

## Good to know

- Default precision is fp64 on CPU, fp32 on GPU (matching `server.py`); pass `--dtype float64` to match the browser exactly.
- Parameter cap: `p ≤ 10,000,000` (memory is `O(p)`).
- CIFAR needs the raw pickle batches — point `--cifar-dir` at the same directory `server.py` uses.
