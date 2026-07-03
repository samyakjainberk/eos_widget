# Progressive Sharpening & Edge of Stability

An interactive tool for watching **how a neural network's loss landscape sharpens as it trains** — and for
**checking whether simple theories can predict that sharpening ahead of time**.

You train a small network with full-batch gradient descent. The page plots, step by step in your browser,
how curved the loss surface is right now — and, next to it, each theory's *forecast* of where the curvature
is going.

> **The idea in a nutshell.** As a network trains, its loss surface gets *sharper* — the largest eigenvalue
> of the loss Hessian grows. Plain gradient descent stays stable only while that sharpness is below
> `2 / learning_rate`. So training pushes the sharpness right up to that limit and then hovers there. The
> climb is **progressive sharpening**; the hovering is the **edge of stability**. This tool lets you watch
> both happen live and score how well the theory calls them.

---

## 🚀 Run it

You need Python and, ideally, an NVIDIA GPU. Three steps:

```bash
pip install torch numpy            # 1. install once
python3 server.py --port 8756      # 2. start the server (auto-picks a free GPU)
                                   # 3. open http://localhost:8756/ and click "Run"
```

Opening **`http://localhost:8756/`** takes you straight to the prediction widget. Choose a dataset from the
dropdown and press **Run** — that's it. You do **not** need to add `/prediction` to the address; the plain
URL already opens it.

**Variations**

| You want… | Command |
|---|---|
| A specific GPU | `python3 server.py --device cuda:0 --port 8756` |
| CPU only (slower, higher precision) | `python3 server.py --device cpu --port 8756` |
| Several GPUs, many browser tabs at once | `python3 server.py --devices all --port 8756` |

In the multi-GPU mode each browser tab is automatically sent to whichever GPU is least busy. The built-in
datasets need no downloads.

**On a remote machine?** Forward the port to your laptop and open the same address:

```bash
ssh -N -L 8756:localhost:8756 you@your-gpu-box
```

> **Seeing a different, busier page at `/` instead of the prediction widget?** Your copy is out of date —
> run `git pull` and hard-refresh the browser.

---

## 🔮 What you're looking at

The page trains a network live and overlays **forecasts** on the real measurements. Each forecast is a
theory's guess at a future quantity, drawn on top of what actually happens — so you can literally see how
close the guess is.

| Section | The forecast | Checked against |
|---|---|---|
| **Prediction 1 / 2 / 2b** | A quick sweep over many `(learning-rate, init-scale)` pairs: an early sign-flip time, and whether early-training signs line up with the loss curvature and the Jacobian's growth | The real sign-flip time and curvature |
| **Prediction 3** | How the error (residual) shrinks after a chosen point — using a frozen Jacobian and a step-updated one | The real error over time |
| **Prediction 4** | How the network's NTK curvature grows from a chosen freeze point | The real top-3 NTK eigenvalues and their directions |
| **Prediction 5.1 / 5.2** | The total curvature (trace) and the peak curvature (sharpness), going forward | The real trace and sharpness |

Good to know:

- **Predictions 3 and 4 run on their own.** Each is seeded from a single snapshot and then propagates
  forward using no live data. You can let them re-sync to the real run every so often (a *"recalc from live
  every N steps"* control; set it to `0` to freeze once and never re-sync). They match the real run closely
  *before* the edge of stability and then drift once past it — the plots cap the drawn value at 20× the real
  one, so a runaway curve stays on screen instead of blowing up the axis.
- **Below the forecasts** sit seven raw-signal panels (loss, sharpness, residual, and several curvature
  spectra) — the measurements the forecasts are built from.
- **Toggles** at the top let you hide any panel you don't need.

For the exact formulas behind each forecast, see **[eos_widget_prediction/README.md](eos_widget_prediction/README.md)**.

---

## 🎛️ Datasets

Pick one from the dropdown; each preset sets sensible defaults (learning rate, init scale, sample count) for
its size.

- **synthetic** — a tiny built-in problem, nothing to download
- **Chebyshev / Chebyshev-2** — the classic edge-of-stability toy (Cohen et al.)
- **k-sparse parity**, **angle-pair** — small structured tasks
- **CIFAR-2 / MNIST-2** — 2-class versions that predict a single ±1 number
- **CIFAR-10 / MNIST** — the full 10-class image tasks

> CIFAR and MNIST need their data files. Put them under `data/` or pass `--cifar-dir <folder>`. The images
> are shrunk to about 500 numbers each so the curvature math stays fast.

---

## 🧩 Three versions, one core

The project ships the same math three ways. They share one **matrix-free** curvature engine (it never builds
a full Hessian — it only multiplies by one) and the same random-number generator, so **for a fixed seed the
training run is identical in all three.**

| Version | What it is | File |
|---|---|---|
| **GPU server** | the widget above — PyTorch, real datasets, larger nets | `server.py` |
| **Browser-only** | a smaller version that runs entirely in JavaScript, no server | `index.html` |
| **`eos_lab/`** | a Python package for scripts and cluster jobs | `eos_lab/` |

---

## 🗺️ Repo map

```
server.py                the GPU server; opens the prediction widget at http://localhost:<port>/
index.html               the no-server, browser-only widget
eos_widget_prediction/   the prediction-widget page + a detailed README
eos_lab/                 Python package for headless / cluster runs
capture_run.py           save a run to a .json the widget can reload offline
run_serve_localhost.sh   the author's SLURM launch script (specific to their cluster)
data/                    CIFAR / MNIST files (not in git; point --cifar-dir here)
```

Package internals: **[eos_lab/README.md](eos_lab/README.md)**.
