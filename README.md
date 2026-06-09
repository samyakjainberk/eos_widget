# Progressive Sharpening & Edge of Stability

An interactive companion to ***Understanding Progressive Sharpening***: it trains a small network with
full-batch gradient descent and visualizes **progressive sharpening (PS)** and the **edge of stability
(EoS)** live, with every panel tied to the paper's equations. Three interchangeable front-ends share one
matrix-free numerical core (Hessian-vector products + Lanczos — no `p×p` matrix is ever formed, so it
scales to multi-million-parameter nets):

| Front-end | What it is | File |
|---|---|---|
| **Browser widget** | zero-backend; the synthetic MLP runs entirely in JS | `index.html` |
| **GPU backend** | same UI, compute in PyTorch streamed over SSE; adds real datasets & big nets | `server.py` |
| **`eos_lab/`** | headless Python package — same diagnostics for scripted / SLURM runs | `eos_lab/` |

## The theory

We train $f(x)\in\mathbb{R}^{d_{\text{out}}}$ ($\theta\in\mathbb{R}^p$) with full-batch GD,
$\theta_{t+1}=\theta_t-\eta\nabla_\theta\mathcal L$. **Sharpness** $\sigma_1=\lambda_{\max}(\nabla^2_\theta\mathcal L)$
rises during training (*progressive sharpening*) until it pins near $2/\eta$ at the **edge of stability**, where
the loss oscillates with period 2.

**Symbols.** $J=\tfrac{df}{d\theta}$ (function Jacobian), $Q=\tfrac{d^2f}{d\theta^2}$ (function Hessian),
$r=y-f$ (residual), $p=\lVert r\rVert$, $\eta$ (learning rate), $N$ (number of samples), $d_n=Nd_{\text{out}}$.
For MSE the loss Hessian splits as $\nabla^2_\theta\mathcal L=G+S$ with the **Gauss–Newton**
$G=\tfrac1N\sum_i\nabla f_i\nabla f_i^\top$ and **residual term** $S=\tfrac1N\sum_i(f_i-y_i)\nabla^2 f_i$.
The **NTK** is $K=JJ^\top$ with eigenpairs $(\sigma_i,u_i)$; via the SVD $J=\sum_j\sqrt{\sigma_j}u_jv_j^\top$,
the $u_i$ / $v_i$ are the NTK left / right singular vectors and $\sigma_1$ the top NTK eigenvalue (the §9 target).
$Q[u]=\sum_a u_a\nabla^2 f_a$ is the function Hessian weighted by $u$, and the function-Hessian tensor factors as
$Q=\sum_i\gamma_ix_iy_i^\top$ — reshaping and eigendecomposing $x_i$ gives eigenvectors $z_{ij}$ and eigenvalues
$\tau_{ij}$, while $y_i$ are its singular vectors.

**Predictive model (Eq. 1).** Over a window, freeze $J,Q$ at $\theta_t$ and approximate

$$f_{\theta_{t+\Delta t}}=f_{\theta_t}+J_{\theta_t}^{\top}\Delta\theta+\Delta\theta^{\top}Q_{\theta_t}\Delta\theta \qquad Q_{\theta_{t+\Delta t}}=Q_{\theta_t}$$

**Single sample (Eq. 13).** With $Q=\sum_i\sigma_iu_iu_i^\top$, propagating $J_{t+1}=(I+\eta r_tQ)J_t$
(so $\sigma_1=\lVert J\rVert^2$) gives, over $s$ steps,

$$(\Delta\mathrm{NTK})_{t\to t+s}=2\eta\prod_{\tau=t}^{t+s}\sum_i (J^\top u_i)^2(r_\tau\sigma_i)(1+\eta r_\tau\sigma_i)^2$$

The NTK grows ($\Rightarrow$ sharpening) when $r\sigma_i>0$ and shrinks when $r\sigma_i<0$, i.e. as long as the
residual $r$ keeps its sign.

**Multiple samples.** Continuously (Eq. 21), the top NTK eigenvalue obeys
$\dot\sigma_1=2\eta\sqrt{\sigma_1}v_1^\top Q[u_1]r^\top J$. Discretely, assuming the residual is aligned
with the top NTK mode (Eq. 27),

$$\sigma_{1,t+s}=\sigma_{1,t}\prod_{k=0}^{s-1}\Big[1+2\eta\sum_{ij}\gamma_i\tau_{ij}(v_{1,t+k}^\top z_{ij})^2(y_i^\top u_{1,t+k})p_{t+k}\Big]$$

where $v_1^\top z_{ij}$ is the alignment of the top Gauss–Newton eigenvector with the function-Hessian tensor and
$y_i^\top u_1$ that of the top NTK eigenvector. Relaxing "aligned with the top mode" to "the residual's
projection onto the top-$|T|$ NTK modes keeps its sign" gives Eq. 29:

$$\sigma_{1,t+s}=\sigma_{1,t}\prod_{k=0}^{s-1}\Big[1+2\eta\sum_{v\in T}\tfrac{\sqrt{\sigma_v}}{\sqrt{\sigma_1}}\sum_{ij}\gamma_i\tau_{ij}(v_{1,t+k}^\top z_{ij})(v_{v,t+k}^\top z_{ij})(y_i^\top u_{1,t+k})p_{t+k}\Big]$$

Progressive sharpening is predicted when the bracketed sum is $>0$ on average; when the (projected) residuals
oscillate and flip sign the growth turns to decay — the cyclic **self-stabilization** at the edge of stability.
Panel §9 overlays each prediction (Eqs. 13 / 21 / 27 / 29) on the measured $\sigma_1$.

## Panels

| § | Panel | Needs |
|---|---|---|
| **1** | loss, sharpness vs `2/η`, residual, spectral edges of `H` | always |
| **2 / 3** | top-/bottom-`n` eigenvalues of `H`, loss Hessian, `G`, `S` | always |
| **4** | `J` projected onto eigenvectors of `H` | always |
| **5** | SLQ spectral density (log) of `H`, `G`, `S`, loss Hessian | always |
| **6** | eigenspace rotation of `H` (principal angles of the ± subspaces) | always |
| **7a / 7 / 8** | NTK alignment; multi-sample NTK + function-Hessian SVD; `vec(J)` onto FH-reshape singular vecs | multi-sample · MSE or CE |
| **4b / 4c** | `J·r` onto `Q[u₁]` eigvecs; `Q[u₁]·(J·r)` onto Gauss–Newton | multi-sample · MSE or CE |
| **4d** | per-residual-sign-group projections | multi-sample · MSE |
| **9** | predicted vs actual sharpness `σ₁` (Eq. 13 single-sample; Eq. 21/27/29 multi-sample) | MSE · single or multi |

A checkbox per section toggles its computation. §7a/§7/§8/§4b/§4c use the *function* NTK `Jᵤ·Jᵤᵀ` and the
generic residual `r = −∂L/∂z` (MSE: `y−f`, CE: `softmax(z)−onehot`), so they work for **cross-entropy** too;
they only need the explicit `M×p` Jacobian (`M = N·d_out`) to be feasible — skipped (stamped *"too large"*)
for e.g. OpenWebText, and empty for a single sample. Only **§4d** (its sign-groups need a scalar residual)
and **§9** (the squared-loss σ₁ recursion) are MSE-only. Any panel a run can't fill is stamped *"n/a"*.

## Datasets · architectures · loss · init

Set from the dropdowns (real-data / conv / transformer runs use the GPU backend):

- **Datasets** — `synthetic` (in-browser MLP), **CIFAR-10**, **sorting** (sort a sequence; MSE),
  **OpenWebText** (GPT-2-BPE next-token LM; cross-entropy, minibatched).
- **Architectures** — `MLP`, `CNN`, **VGG11** (`chmul` scales channels, up to ~9.4M params),
  **mini-GPT** (sorting regressor, or a tied-embedding token-LM for OpenWebText).
- **Loss** — `MSE` (all panels) or `cross-entropy` (keeps §4/§6; residual-NTK & theory panels n/a).
- **Init** — `default` (`scale/√fan_in`), `muP`, `Xavier normal`, `Xavier uniform`, or `custom` (Gaussian
  std); `init_scale` is the scale/gain/std for the chosen scheme.

Below the main widget, the **quadratic / linear surrogate** sections train the real model and a
frozen-window Taylor surrogate (Eq. 1, with/without the curvature term) in lockstep and overlay their
loss, residuals, eigenvalues, §9 theory and projections — actual solid, surrogate dashed.

## Run

```bash
# Browser widget (zero backend; also deploys to GitHub Pages as-is):
python3 -m http.server 8000                 # open http://localhost:8000/

# GPU backend (only dependency is torch; serves the same index.html over SSE):
python3 server.py --device cuda:0 --port 8756 --cifar-dir /path/to/cifar-10-batches-py
#   then in the page set Compute → "GPU backend (server.py)"

# Headless (eos_lab):
python -m eos_lab.cli --list-presets
python -m eos_lab.cli --preset owt_gpt --device cuda:0         # → results.pt + section*.png
python -m eos_lab.cli --preset cifar_mlp --set initscheme=xavier_uniform --set lr=0.03
python -m eos_lab.owt_prepare                                  # build the OpenWebText token shard
```

Data + init use the same `mulberry32` RNG across all three, so the GD trajectory matches for a given seed.

## Layout

```
index.html      browser widget (also served by server.py)
server.py       GPU backend — HTTP + SSE, PyTorch
eos_lab/        headless package: config (presets) · models · data · train · diagnostics · plots · cli · slurm/
data/           cifar-10-batches-py/ and owt/   (git-ignored; regenerable)
```

Run outputs and datasets are git-ignored — rebuild the OWT shard with `python -m eos_lab.owt_prepare` and
point `--cifar-dir` at the raw CIFAR `data_batch_*` files.
