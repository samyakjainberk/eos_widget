# Progressive Sharpening & Edge of Stability

Train a small net with full-batch GD and **watch the curvature climb to the edge of stability, live** —
with the paper's theory overlaid on what the net actually does. The **prediction widget** (`/prediction`)
turns those diagnostics into **forecasts** and scores each one against the real run.

> **As a net trains, its loss surface gets sharper** (top Hessian eigenvalue climbs). GD is stable only
> while sharpness stays below `2/η`, so training drives itself to that line and hovers — the climb is
> *progressive sharpening*, the hovering is the *edge of stability*.

---

## 🚀 Run the prediction widget on your GPU

**Any machine with an NVIDIA GPU → one server → one URL → press Run.**

```bash
pip install torch numpy                          # 1 · install (once)
python3 server.py --device cuda:0 --port 8756    # 2 · start the backend on your GPU
#                                                  3 · open in a browser ↓
#     http://localhost:8756/
```

Opening the port (**`http://localhost:8756/`**) serves the **prediction widget** directly — the
widget **defaults to the GPU backend**, so just open the URL and press **Run**. (The `/prediction`
path works too.)

> **Running on a remote GPU box?** Forward the port to your laptop, then open the same URL:
> ```bash
> ssh -N -L 8756:localhost:8756  you@gpu-host
> ```

- **CPU only** (fp64, slower): `python3 server.py --device cpu --port 8756`
- **Multi-GPU pool** (concurrent tabs auto-spread across GPUs): `python3 server.py --devices all --port 8756`
- Pick the dataset / architecture / size from the **preset dropdown**; press **Run**.

---

## 🔮 What the prediction widget shows

Seven base panels (§1–§7) feed the ★ **forecasts**, each scored against the actual run:

| Panel | Forecast | vs. the actual run |
|---|---|---|
| **1 / 2 / 2b** | `(lr, init)` **sweep** (`Run sweep`) | sign-change timing `t*`; `sign(d)` vs. loss-curvature at iter `t`; vs. start-of-training `‖J‖` trend |
| **3** | frozen-`J` residual (1st + 2nd order) **and evolving-`J`** — exact `Ĵ += (η/N)·Q·(Ĵᵀr̂)`, `J` updated as a step-function every `k` iters | `‖r‖` decay and its direction |
| **4** | frozen- vs. **evolving-`Qr`** — `Ĵ += (η/N)·M_r·Ĵ` every step, `rQ` fixed per `s`-iter window — NTK propagation from `t₀` | top-3 NTK eigenvalues + eigenvectors |
| **5.1 / 5.2** | quad / cubic × live / self | `Tr(NTK)` and sharpness `σ₁` vs. the actual `Tr(∇²L)`, `λmax` |

- **Prediction-3 & 4 are fully self-contained:** seeded once at the freeze point, then propagated with
  **no further information from the live run**. The evolving curves follow the *true* recurrence — Pred-3
  the exact Jacobian update `Ĵ += (η/N)·Q·(Ĵᵀr̂)` (step-function, `J` refreshed every `k` iters), Pred-4
  the residual-weighted-Hessian power iteration `Ĵ += (η/N)·M_r·Ĵ` (`rQ` held fixed per `s`-iter window).
  Both **track the actual well before the edge of stability, then diverge past it** (the linear / `M_r`
  recurrence is expansive there) — the plots cap the displayed value at 20× the actual so a divergence
  stays readable rather than blowing the axis.
- **§4** is the SLQ spectral density of the function Hessian `H`, Gauss-Newton `G`, residual term `S`, and
  the full loss Hessian `∇²L` — refreshed every *SLQ-every* iterations (a control in the bar).
- Base-panel and forecast toggles live in the top bar; turn one off to hide it.

Full panel-by-panel docs live in **[eos_widget_prediction/README.md](eos_widget_prediction/README.md)**.

---

## 🎛️ Datasets

Each preset auto-tunes `lr` / `init` / `N` and the sections that fit that size.

`synthetic` (in-browser MLP) · **CIFAR-10 / MNIST** (10-class) · **CIFAR-2 / MNIST-2** (2-class **scalar**
±1, `d_out=1`) · **Chebyshev / Chebyshev-2** (Cohen et al. EoS toy) · **k-sparse parity** · **angle-pair**
(single- or multi-sample).

> CIFAR/MNIST images are compressed to **≈500 input dims** (`3×13×13`, a fixed average-pool) so the
> Jacobian / Hessian-vector work stays feasible; the CNN and VGG stacks convolve the smaller image.

---

## 🧩 Three backends, one core

All three share one **matrix-free** numerical core (Hessian-vector products + Lanczos — no `p×p` matrix is
ever formed), and use the same `mulberry32` RNG, so at a fixed seed the **GD trajectory is bit-identical**
everywhere.

| Backend | What it is | File |
|---|---|---|
| **GPU backend** | the widget above — PyTorch over SSE; real datasets + big nets | `server.py` |
| **Browser widget** | zero-backend synthetic MLP that runs entirely in JS | `index.html` |
| **`eos_lab/`** | headless Python package for scripted / SLURM runs | `eos_lab/` |

---

## 🗺️ Repo map

```
server.py                GPU backend — HTTP + SSE (also serves the /prediction widget)
index.html               zero-backend browser widget
eos_widget_prediction/   the /prediction forecast widget (index_prediction.html + README)
eos_lab/                 headless package: config · models · data · diagnostics · prediction · cli
capture_run.py           headless capture → a self-contained .json the widget reloads offline
run_serve_localhost.sh   serve the GPU backend on SLURM + a reverse SSH tunnel (author's cluster)
data/                    cifar-10-batches-py/, mnist/   (git-ignored; point --cifar-dir at the CIFAR files)
```

See **[eos_lab/README.md](eos_lab/README.md)** for the package internals.
