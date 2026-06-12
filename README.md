# Progressive Sharpening & Edge of Stability

An interactive companion to ***Understanding Progressive Sharpening***: it trains a small network with
full-batch gradient descent and visualizes **progressive sharpening (PS)** and the **edge of stability
(EoS)** live, with every panel tied to the paper's equations.

**The gist.** As a network trains, its loss surface keeps getting *sharper* — the curvature (the top
eigenvalue of the loss Hessian) climbs. Gradient descent is only stable while that sharpness stays below
`2/η`, so training drives itself right up to that line and then hovers there. The climb is *progressive sharpening*; the hovering is the *edge of stability*. This
tool runs that experiment in front of you and overlays the paper's predictions so you can see how well they
match what the network actually does. (If you just want to watch it, skip the math below and press **Run**.)

Three interchangeable front-ends share one matrix-free numerical core (Hessian-vector products + Lanczos —
no `p×p` matrix is ever formed, so it scales to multi-million-parameter nets):

| Front-end | What it is | File |
|---|---|---|
| **Browser widget** | zero-backend; the synthetic MLP runs entirely in JS | `index.html` |
| **GPU backend** | same UI, compute in PyTorch streamed over SSE; adds real datasets & big nets | `server.py` |
| **`eos_lab/`** | headless Python package — same diagnostics for scripted / SLURM runs | `eos_lab/` |

## The theory

This section collects the symbols and equations each panel is tied to. It's reference material — handy if
you're following along with the paper, skippable if you just want to drive the tool.

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
$\dot\sigma_1=2\eta\sqrt{\sigma_1}\,v_1^\top Q[u_1]\,(J^\top r)$, where $J^\top r=\sum_j\sqrt{\sigma_j}\,(r\cdot u_j)\,v_j$.
The paper closes this two ways.

**Eq. 22** assumes the residual aligns with the top NTK mode ($\hat r\simeq u_1$, so $r\approx\lVert r\rVert\,u_1$) and
keeps only that mode:

$$\dot\sigma_1\approx 2\eta\,\lVert r\rVert\,\sigma_1\,v_1^\top Q[u_1]v_1,\qquad
\sigma_{1,t+1}=\sigma_{1,t}\big[1+2\eta\,\lVert r\rVert\,v_1^\top Q[u_1]v_1\big].$$

**Eq. 23** drops that full-alignment assumption and decomposes $J^\top r$ over the top-$|T|$ NTK modes (so it
recovers Eq. 22 at $|T|=1$, but with the projection $r\cdot u_1$ in place of $\lVert r\rVert$):

$$\dot\sigma_1\approx 2\eta\sqrt{\sigma_1}\sum_{k\in T}\sqrt{\sigma_k}\,(r\cdot u_k)\,v_1^\top Q[u_1]v_k,\qquad
\sigma_{1,t+1}=\sigma_{1,t}\Big[1+2\eta\sum_{k\in T}\tfrac{\sqrt{\sigma_k}}{\sqrt{\sigma_1}}\,(r\cdot u_k)\,v_1^\top Q[u_1]v_k\Big].$$

The **col-3 (Eq. 22)** panel uses $p_t=r\cdot u_1$ — the residual's projection onto $u_1$, i.e. the $|T|=1$ case of
Eq. 23 — rather than the literal $\lVert r\rVert$: the two agree under the alignment assumption, but the projection
tracks the actual residual ($\lVert r\rVert$ overshoots when it isn't aligned). Both Eq. 22 and **col-4 (Eq. 23)**
evaluate the bilinear $v_1^\top Q[u_1]v_k$ **exactly**, via a frozen-$Q$ Hessian-vector product ($Q[u_1]v_1$ once per
window, then dotted with each $v_k$).

Relaxing "aligned with the top mode" to "the residual's projection onto the top-$|T|$ NTK modes keeps its sign,"
and evaluating the bilinear through the $\sum_{ij}\gamma_i\tau_{ij}\dots$ **eigendecomposition of $Q$** (which assumes
the reshaped $x_i$ are symmetric), gives **Eq. 29** — the **col-5** panel:

$$\sigma_{1,t+s}=\sigma_{1,t}\prod_{k=0}^{s-1}\Big[1+2\eta\sum_{v\in T}\tfrac{\sqrt{\sigma_v}}{\sqrt{\sigma_1}}\sum_{ij}\gamma_i\tau_{ij}(v_{1,t+k}^\top z_{ij})(v_{v,t+k}^\top z_{ij})(y_i^\top u_{1,t+k})\,(r\cdot u_v)\Big]$$

So **col-4 (Eq. 23)** and **col-5 (Eq. 29)** are the *same* top-$|T|$-mode sum; they differ only in how
$v_1^\top Q[u_1]v_k$ is evaluated — directly (exact) vs through the $Q$-eigendecomposition — and the gap between
them measures that approximation error.

Progressive sharpening is predicted when the bracketed sum is $>0$ on average; when the (projected) residuals
oscillate and flip sign the growth turns to decay — the cyclic **self-stabilization** at the edge of stability.
Panel §9 overlays each prediction (Eqs. 13 / 21 / 22 / 23 / 29) on the measured $\sigma_1$ (the top
Gauss–Newton / NTK eigenvalue). Two companion panels reuse the same predictions:
**§9b** adds the dropped second-order term $\lVert\Delta J^\top u_1\rVert^2=(\eta/N)^2\lVert q_u\rVert^2\ge 0$
(single-sample $\lVert\Delta J\rVert^2$) — an always-non-negative sharpening floor the first-order recursion omits;
**§9c** compares the predictions against the **full loss-Hessian sharpness** $\lambda_{\max}(\nabla^2\mathcal L)=\lambda_{\max}(G+S)$
instead of the Gauss–Newton edge, so the gap is exactly the residual term $S$.
**§9d / §9d-c** repeat §9 and §9c but compute the residual *from scratch* with the frozen quadratic model —
$r_q=Y-f_{\text{quad}}(\theta_0+\Delta\theta)$, $f_{\text{quad}}=f_0+J_0\Delta\theta+\tfrac12\Delta\theta^\top Q\Delta\theta$, with
$\Delta\theta$ advanced by the quadratic model's own gradient descent — a fully closed prediction that uses no live-run
input inside a window; it coincides with §9 at each window start ($\Delta\theta=0$) and diverges by exactly the quadratic-approximation error.

**§10 (cubic approximation, off by default)** is the **cubic** model (paper §6): every `cubic window` steps it freezes
$\theta_0$, $J_0$, $Q_0$ and the third-derivative tensor $T$, then propagates **both** $J$ and $Q$ exactly and predicts the
NTK-$\sigma_1$ change — **Eq. 47** (single sample) and **Eq. 51** (multi sample), each **with** and **without** the PSD term —
plus Eq. 51 *without* the $\eta^2$ cubic term (the quadratic recursion). Two panels overlay these on the empirical NTK
$\sigma_1$ (panel 1, + the window drift $\lVert\Delta Q\rVert/\lVert Q_0\rVert\times100$) and on the full-Hessian
$\lambda_{\max}$ (panel 2, + $\lVert\Delta J\rVert/\lVert J_0\rVert\times100$). The same two panels appear in every section. It is
the heaviest panel ($Q$ is propagated exactly, $O(\text{window}^2)$ HVPs/window), hence off by default; a fourth surrogate
section trains the full cubic Taylor model $f_0+J_0\Delta\theta+\tfrac12\Delta\theta^\top Q\Delta\theta+\tfrac16T[\Delta\theta,\Delta\theta,\Delta\theta]$ against the actual model, exactly like the quadratic/linear surrogate sections.

## Panels

| § | Panel | Needs |
|---|---|---|
| **1** | loss, sharpness vs `2/η`, residual, spectral edges of `H` | always |
| **2 / 3** | top-/bottom-`n` eigenvalues of `H`, loss Hessian, `G`, `S` | always |
| **4** | `J` projected onto eigenvectors of `H` | always |
| **5** | SLQ spectral density (log) of `H`, `G`, `S`, loss Hessian | always |
| **6** | eigenspace rotation of `H` (principal angles of the ± subspaces) | always |
| **7a / 7 / 8** | NTK alignment (+ the per-step products `Δλ·Δ⟨r,vₖ⟩`, synchronous & 1-step-lagged, with running averages); multi-sample NTK + function-Hessian SVD; `vec(J)` onto FH-reshape singular vecs | multi-sample · MSE or CE |
| **4b / 4c** | `J·r` onto `Q[u₁]` eigvecs; `Q[u₁]·(J·r)` onto Gauss–Newton | multi-sample · MSE or CE |
| **4d** | per-residual-sign-group projections | multi-sample · MSE |
| **9** | predicted vs actual sharpness `σ₁` (Eq. 13 single-sample; Eq. 21/22/23/29 multi-sample) | MSE · single or multi |
| **9b** | the same predictions **+ the 2nd-order PSD term** `‖ΔJᵀu₁‖²` vs actual `σ₁` | MSE · single or multi |
| **9c** | the same predictions vs the **full-Hessian sharpness** `λmax(∇²L)` (own toggle) | MSE · single or multi |
| **9d / 9d-c** | §9 / §9c but with the residual **self-computed by the quadratic model** (closed loop), vs actual `σ₁` / the full-Hessian sharpness (own toggles) | MSE · single or multi |
| **10** | **cubic** approximation — Eq. 47/51 σ₁ predictions (±PSD, and Eq. 51 without the η² term) with exact `J`&`Q` propagation, vs the NTK `σ₁` and the full-Hessian `λmax`, + the `‖ΔQ‖`/`‖ΔJ‖` window drift (off by default — heaviest) | MSE · single (Eq. 47) or multi (Eq. 51) |

A checkbox per section toggles its computation. §7a/§7/§8/§4b/§4c use the *function* NTK `Jᵤ·Jᵤᵀ` and the
generic residual `r = −∂L/∂z` (MSE: `y−f`, CE: `softmax(z)−onehot`), so they work for **cross-entropy** too;
they only need the explicit `M×p` Jacobian (`M = N·d_out`) to be feasible — skipped (stamped *"too large"*)
for e.g. OpenWebText, and empty for a single sample. Only **§4d** (its sign-groups need a scalar residual)
and **§9** (the squared-loss σ₁ recursion) are MSE-only. Any panel a run can't fill is stamped *"n/a"*.

The cheap core (loss/sharpness/eigenvalues/§9 theory/§7a) is computed every diagnostic tick; the heaviest
slowly-varying panels — §7's FH-eigenvector projections and §8 — are computed every **`heavy every`** ticks
(default 4) and §5 SLQ ~50×/run, so a run stays responsive even with everything on. Set `heavy every` to 1 to
compute every panel every step.

## Datasets · architectures · loss · init

Set from the dropdowns (real-data / conv / transformer runs use the GPU backend):

- **Datasets** — `synthetic` (in-browser MLP), **CIFAR-10**, **sorting** (sort a sequence; MSE),
  **OpenWebText** (GPT-2-BPE next-token LM; cross-entropy *or* MSE, minibatched — under MSE the
  logits are regressed toward the one-hot next token, normalised per token), and **Chebyshev**
  (Cohen et al. EoS toy task: `nsamp` points evenly spaced on [-1,1] labeled by the Chebyshev
  polynomial `T_degree`; MSE on a 6-hidden-layer width-50 tanh MLP, ≈12.9k params.
  `degree` is set in the hyperparameter panel; the preset sharpens to the `2/η` edge of stability).
- **Architectures** — `MLP`, `CNN`, **VGG11** (`chmul` scales channels, up to ~9.4M params),
  **mini-GPT** (sorting regressor, or a tied-embedding token-LM for OpenWebText).
- **Loss** — `MSE` (all panels) or `cross-entropy` (keeps §4/§6; residual-NTK & theory panels n/a).
- **Init** — `default` (`scale/√fan_in`), `muP`, `Xavier normal`, `Xavier uniform`, or `custom` (Gaussian
  std); `init_scale` is the scale/gain/std for the chosen scheme.

Below the main widget, the **quadratic / linear surrogate** sections train the real model and a
frozen-window Taylor surrogate (Eq. 1, with/without the curvature term) in lockstep and overlay their
loss, residuals, eigenvalues, the §9 / §9b / §9c / §9d σ₁-prediction panels (same Eq. 13/21/22/23/29 columns,
with the PSD term and the full-Hessian-sharpness comparison), and projections — actual solid, surrogate
dashed. They default
to the **EoS-demo** preset (a small tanh MLP whose loss visibly trains to the edge of stability), so the
first loss panel shows real evolution: the quadratic surrogate tracks the actual within each window, and
the linear (NTK) surrogate diverges once the frozen sharpness passes `2/η` — expected at EoS. (The wide
`width=100` presets keep the loss near-flat; their dynamics live in the sharpness panel, not the loss.)

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
