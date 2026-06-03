# Progressive Sharpening / Edge-of-Stability playground

A single-file, **zero-backend** interactive widget that trains a small scalar-output
network full-batch with gradient descent and visualizes progressive sharpening (PS)
and the edge of stability (EoS) **live, as training runs**. Everything — forward/backprop,
finite-difference Hessian–vector products, the sharpness, and the function-Hessian
eigenvectors — runs client-side in JavaScript. The only external dependencies are
[Plotly.js](https://plotly.com/javascript/) and [KaTeX](https://katex.org/) from a CDN.

Because it is a static `index.html`, it deploys to **GitHub Pages** with no build step.

## What it shows

The network is a chain of dense layers with a `d_out`-dimensional output, trained with GD on
`number_of_samples` points. Let `f(x)` be the output, `J = ∇_θ (Σ_i f(x_i))`, and
`H = ∇²_θ (Σ_i f(x_i))` the **function Hessian**. For MSE loss the loss Hessian decomposes as
`∇²L = G + S` with the **Gauss-Newton** `G = (1/N)Σ ∇f_i ∇f_iᵀ` (PSD) and the **residual term**
`S = (1/N)Σ (f_i − y_i) ∇²f_i`. Plots update point-by-point during the run.

Every section is a row of **four** plots. Hover any step-axis plot for a **synchronized dotted
cursor** drawn across all of them (the SLQ plots use an eigenvalue x-axis, so they're excluded).

1. **Loss, sharpness, residual, and spectral edges of H.** *Sharpness* = top (largest-algebraic)
   eigenvalue of the loss Hessian; the dashed line is the EoS threshold `2/lr`. *Residual* is the
   **signed** `y − f(x)` per output component (zero line). Last panel: the largest & smallest
   eigenvalue of the function Hessian `H`.
2. **Eigenvalue evolution — top-n** for `H`, the loss Hessian, Gauss-Newton `G`, and the residual
   term `S` (`λ_max(loss Hessian) =` sharpness, `loss Hessian = G + S`).
3. **Eigenvalue evolution — bottom-n** for the same four matrices.
4. **Eigen-spectrum (SLQ).** Stochastic Lanczos Quadrature estimate of the spectral density
   (density of eigenvalues), refreshed live, for `H`, `G`, `S`, and the full loss Hessian.
5. **Eigenspace rotation of H.** Principal angles (max & mean, degrees) between the dominant
   eigenspace at consecutive steps — separately for the subspace capturing **energy %** of the
   positive vs negative eigenvalue energy. This is how to see *how the function Hessian's
   eigenvectors are changing direction*.

Order: §1, §2 (top-n), §3 (bottom-n), §4 (J projections), §5 (SLQ), §6 (eigenspace rotation).

Eigenvector signs are tracked across GD steps by similarity to the previous step. The energy-%
subspaces are built from the top/bottom Ritz pairs (energy normalized by the captured part, up to
~10 each) — for a spread spectrum that cap can saturate, so lower **energy %** for a smaller,
more stable subspace.

**Panels toggle bar** — a checkbox per section (plus a **G & S spectra** sub-toggle); turning a
panel off **skips its computation** (not just hides it), so you can run only what you need. SLQ
(§5) and eigenspace rotation (§6) are the most expensive.

## Scaling — up to 100,000 parameters

The sharpness and the top-/bottom-`n` eigenvectors of `H` are computed **matrix-free with
Lanczos** (Hessian–vector products only) plus power-style iteration — no `p × p` matrix is
ever formed. This scales to large networks: the parameter count is capped at **`p ≤ 100,000`**.
Each analysis step needs many HVPs, so for large `p` raise **`eig every`** to keep the live
animation responsive. `n` is clamped to `⌊p/2⌋`.

## Presets

Pick a preset from the dropdown (it auto-runs).

- **Wide deep-linear net — single sample (fixed input)** *(default).*
  `linear, depth=4, width=100, bias=off, in_dim=10, fixed input x=1`, one sample (`p ≈ 31,000`).
  A large net to exercise the matrix-free machinery; **SLQ (§5) is off by default** at this size
  (re-enable it via the Panels bar if you want the spectra). Presets also set the panel toggles.

- **Multi-sample tanh MLP — progressive sharpening → EoS.**
  A small tanh net (`width=3, depth=3`) on 40 points in the non-interpolating regime.
  Sharpness climbs, pins at `2/lr`, and the loss settles into a period-2 EoS oscillation;
  `J` projects non-trivially onto both the top and bottom eigenvectors of `H`. All panels on.

Editing any field switches the dropdown to "custom". **Run** (re)starts; **Reset** stops and
clears. A `steps/sec` readout shows live throughput.

## Controls

`preset · depth · width · activation (linear/tanh/elu) · bias · fixed input (x=1) ·
in_dim · out_dim · number_of_samples · target_scale · learning rate · init_scale · steps · seed ·
n (top/bottom eigs) · k (which single eigvec, independent of n) · eig every`.

## Run locally

```bash
cd eos_widget
python3 -m http.server 8000
# then open http://localhost:8000/
```

## Run on the GPU — live plots, server-side compute (`server.py`)

There is a single page (`index.html`) with a **Compute** dropdown:

- **browser (this tab)** *(default — quicker for small/medium nets)*: every HVP / Lanczos / SLQ
  runs in single-threaded JS on the CPU. Zero backend; works on GitHub Pages.
- **GPU backend (server.py)**: the same computation runs in **PyTorch on a GPU** and the per-step
  diagnostics stream back over **Server-Sent Events**; the browser only draws. Use this for big
  nets / long runs the browser can't handle. Needs `server.py` running (the page connects to its
  `/run` endpoint); if it isn't, the page says so — just switch Compute back to browser.

`server.py` is pure standard-library HTTP — no flask/fastapi/websockets — so the only dependency
beyond the page is `torch`. It serves the same `index.html`, so both modes live on one page.
(`index_server.html` now just redirects to `index.html` for old links.)

```bash
cd eos_widget
python3 server.py                 # auto-picks the freest GPU, fp32; serves index.html
python3 server.py --device cuda:2 --port 8756   # pin a GPU / port
# from your laptop, tunnel one port and open it (or use VS Code's PORTS panel):
#   ssh -N -L 8756:localhost:8756 <this-box>
#   open http://localhost:8756/    then set Compute → "GPU backend (server.py)"
```

Flags: `--device cpu|cuda|cuda:N` (default `auto` = freest GPU), `--dtype float32|float64`
(default fp32 on GPU, fp64 on CPU), `--host` / `--port`. `index.html` is untouched and still
works as the zero-backend version; `index_server.html` is the GPU-backed copy (identical UI,
compute swapped for the SSE stream). All the controls, presets, and panel toggles behave the
same; the status line shows the backend device.

**When the GPU actually helps:** only for *large* nets (p in the tens of thousands toward the
100k cap, larger N). For the tiny default presets (the product preset is 6 parameters) a GPU
is slower than CPU due to kernel-launch overhead — but moving from browser-JS to native
PyTorch is a big speedup regardless. Note A6000-class GPUs run fp64 at ~1/32 of fp32, so the
default fp32 is much faster; use `--dtype float64` only if you need to match the browser's
double precision exactly. Data + initialization use the same `mulberry32` RNG as the browser,
so the GD trajectory is identical for a given seed; Lanczos/SLQ probe vectors use torch's RNG
(extreme eigenvalues converge regardless of start vector), so the curves match closely.

## Deploy to GitHub Pages

**Option A — project site from the repo (simplest).**
`index.html` is at the repo root, so just push. On GitHub: **Settings → Pages**, set
**Source = Deploy from a branch**, pick branch `main` and folder `/ (root)`, **Save**.
It publishes at `https://<user>.github.io/<repo>/`.

**Option B — dedicated `docs/` folder.** Copy `index.html` into a top-level `docs/`,
set Pages *Source = branch*, folder `/docs` → serves at `https://<user>.github.io/<repo>/`.

**Option C — standalone site.** Put `index.html` at the root of a `<user>.github.io` repo.

No build/CI is required. The first load fetches Plotly + KaTeX from CDNs, so the browser needs
internet access (to vendor offline, download the libs next to `index.html` and use relative `src`s).

## Notes

- `_preview.py` is a faithful Python/PyTorch port (same mulberry32 RNG, same model, same
  Lanczos) used to validate the numerics and render the `preview_*.png` images. The
  hand-written backprop matches autograd to ~1e-8 (linear) / 1e-11 (tanh); the Lanczos
  top-/bottom-`n` eigenpairs match a full dense eigendecomposition to ~4 decimals.
- The single-sample product preset mirrors `single_sample_eos_oscillation.py`; the widget adds
  one output-weight factor, so widget `depth = D` ↔ a `D + 1`-factor product.
