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

**§10 — cubic approximation** (off by default; paper §6). Freeze $J=\nabla f,\ Q=\nabla^2 f,\ T=\nabla^3 f$ at $\theta_0$ every `cubic window` steps; with GD step $\Delta\theta=\eta\,J^\top r$ ($\eta\!\to\!\eta/N$ multi-sample), propagate $J,Q$ over the window ($T$ frozen):

$$f_{\theta_0+\Delta\theta}=f_0+J^\top\Delta\theta+\tfrac12\Delta\theta^\top Q\,\Delta\theta+\tfrac16 T[\Delta\theta,\Delta\theta,\Delta\theta],\qquad \Delta J=Q\,\Delta\theta+\tfrac12 T[\Delta\theta,\Delta\theta]$$

$$\text{single (Eq.\,47):}\ \ \Delta\mathrm{NTK}=2J^\top\Delta J+\lVert\Delta J\rVert^2,\qquad\quad \text{multi (Eq.\,51):}\ \ \Delta\sigma_1=2\sqrt{\sigma_1}\,v_1^\top(\Delta J^\top u_1)+\lVert\Delta J^\top u_1\rVert^2$$

$(u_1,v_1,\sigma_1)$ is the top NTK singular triple of the propagated $J$. Three curves: **cubic** (full $\Delta J$), **+ PSD** (keeps the $\lVert\cdot\rVert^2$ term), **without $\eta^2$** (drops $\tfrac12 T[\Delta\theta,\Delta\theta]$ from $\Delta J$). Panel 1 compares to the empirical NTK $\sigma_1$ (+ drift $\lVert\Delta Q\rVert/\lVert Q_0\rVert$); panel 2 to the full-Hessian $\lambda_{\max}$ (+ $\lVert\Delta J\rVert/\lVert J_0\rVert$). Heaviest panel ($Q$ propagated exactly, $O(\text{window}^2)$ HVPs); a fourth surrogate section trains the cubic Taylor model $f_{\theta_0+\Delta\theta}$ above against the actual model.

**§20 — residual-weighted function Hessian $M_r$.** The function-Hessian "$S$" term of the mean-loss Hessian,
$M_r=\tfrac1N\sum_k r_k Q_k$ ($Q_k=\nabla^2 f_k$, $r_k$ the residual), reported $\div N$ — the same per-sample scale
as §1's sharpness and §2/§3's eigenvalues. Its **top-$K$ ⊕ bottom-$K$** eigenpairs $(\lambda_i,v_i)$ are extracted
matrix-free by Lanczos ($K$ from the **s28k** box, default 40). Plot 1 is the eigenvalue histogram; plots 2/3/4 overlay
$\lambda_i\,\lvert\langle v_i,u_k\rangle\rvert$ on the bare $\lambda_i$, where $u_k$ is the $k$-th **right-singular
vector** of $J$ (the top-4 Jacobian directions) — how much of each eigenvalue survives projection onto that direction.
The histograms **evolve over training**; a slider scrubs the snapshots. Small $N$ ($M=N\,d_{\text{out}}\le$ `grid3dcap`).

**§21 — residual ↔ spectrum alignment.** Three panels (slider): **(1)** the residual $r$ projected onto the top-$K$
NTK $K=JJ^\top$ eigenvectors ($\sigma_i\div N$-weighted bars); **(2)** $J\!\cdot\!r=\sum_k r_k\nabla f_k$ onto the
top-$K$ ⊕ bottom-$K$ $M_r$ eigenvectors (signed $\lambda\div N$); **(3)** $J\!\cdot\!r$ onto the top-$K$ Gauss–Newton
$G=\tfrac1N J^\top J$ eigenvectors $z_i$ (= the right-singular vectors of $J$, $c_i=\sigma\div N$). $K$ from **s29k**
(default 40). Does the residual concentrate on the leading curvature modes?

**§22 / §23 — quadratic-Taylor surrogate REPLACE mode** (off by default; distinct from the lockstep compare mode below).
Selecting one makes the **whole run** a quadratic-Taylor surrogate: the surrogate trajectory — *not* real GD — drives
§1 (loss/sharpness/residual), §2/§3 (function-Hessian / loss-Hessian / Gauss–Newton=NTK / residual spectra) and §20,
with §4–§21 and §24 turned **off** and a warning banner shown. **§22** freezes $Q$ at the iteration $t$ you set (**s30t**):
real GD reaches $\theta_t$, then GD runs on the frozen-$Q$ quadratic MSE loss. **§23** instead uses a fixed low-rank
**random** symmetric $Q$ (rank **s31r**, spectral norm = the real function-Hessian's $\lambda_{\max}$ at $\theta_0\times$
**s31scale**) trained from $\theta_0$ — a **null model** asking whether the §20 alignment patterns appear under random
curvature. The two are **mutually exclusive** (frozen-Q wins). MSE only, small $N$.

**§24 — 1st/2nd-order $\Delta f$ alignment.** Decomposes the one-GD-step change in each output $f_k$ into a first-order
term $A=JJ^\top r$ (NTK·$r$) and a second-order term $B_k=\tfrac{\eta}{2N}(J\!\cdot\!r)^\top Q_k(J\!\cdot\!r)$, and plots
$\lVert A\rVert$, $\lVert B\rVert$, $\lvert\cos(A,r)\rvert$, $\lvert\cos(B,r)\rvert$ and eigenvalue-weighted projections of
$A,B$ onto the residual $r$ and the top-4 NTK eigenvectors $u_1$–$u_4$ (NTK eigenvalues $\div N$), as a time-series. GPU
backend only, off in surrogate mode, $\text{dataset}\ne$ owt, small $N$.

**§25 — gradient-norm & $d/dt(J\!\cdot\!r)$ evolution.** A single **linear**-axis time-series with five curves:
$\lVert\nabla L\rVert$; $\lVert\dot J\!\cdot\!r\rVert=\lVert M_r\nabla L\rVert$ ($M_r=\sum_k r_kQ_k$);
$\lVert J\!\cdot\!\dot r\rVert=\lVert J^\top J\nabla L\rVert$ (the two product-rule terms of $d/dt(J\!\cdot\!r)$ under
gradient flow $\dot\theta=-\nabla L$); and $\lvert\langle\nabla\lVert J\rVert_F^2,-Q\nabla L\rangle\rvert$,
$\lvert\langle\nabla\lVert J\rVert_F^2,G\nabla L\rangle\rvert$, where $\nabla\lVert J\rVert_F^2=2\sum_k Q_kJ_k$ and
$Q=\tfrac1N\sum_k Q_k$, $G=\tfrac1N J^\top J$ are **both $\div N$** so the two inner products are comparable. GPU backend
only, off in surrogate mode, small $N$.

## Panels

| § | Panel | Needs |
|---|---|---|
| **1** | loss, sharpness vs `2/η`, residual, spectral edges of `H` | always |
| **2 / 3** | top-/bottom-`n` eigenvalues of `H`, loss Hessian, `G`, `S` | always |
| **4** | `J` onto the eigenvectors of `H`, in **3 phases** — phase 1 alignment `\|⟨J,u⟩\|/‖J‖`, phase 2 power-iteration `\|⟨J,u⟩\|·σ`, phase 3 residual-dominance `\|⟨J·r,u⟩\|·σ` — with `σ÷N` (per-sample scale) and the single-sample residual kept signed | always |
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
| **11** | 3D Hessian–NTK grids `T1=JᵢᵀQⱼJₖ`, `T2=uⱼuₖT1`, `T3=rᵢuⱼuₖT1` — per-`(i,j,k)`-class evolution stats + sparsified cuboids (rotating GIFs offline) | multi-sample · small `N` (N³ grid, capped by `grid3dcap`) |
| **12a / 12b** | per-sample function Hessian `Qᵢ=∇²fᵢ`: top/bottom-eigvec **principal angles** + diagonal alignment across samples (12a); per-sample **projection** panels of `∇f` / `J·r` onto each `Qᵢ`'s eigvecs (12b) | multi-sample · small `N` (capped by `sec12ncap`) |
| **13** | residual-weighted curvature `G1/G2/G3 ≈ ∑ JᵢᵀQⱼJₖ·rₖ` vs the exact reference (own toggle) | multi-sample · small `N` |
| **14** | `Tr(ΔNTK)` per-triplet decomposition over the `N³` cube | multi-sample · small `N` |
| **15** | 2nd-difference decomposition of `‖J‖²_F` (=tr NTK) and `σ₁` into theory terms I/II/III (+ IV/V/VI), the matrices A/B, chained-contraction norms, and a per-sample top/bottom projection ratio | MSE · single or multi |
| **16** | standalone **curvature-aligned per-residual-sign optimizer** — from θ₀: a GD warmup, then **project** the positive-residual gradient `g₊` onto the **top-`kdir`** eigvectors of the sample-averaged function Hessian `H̄` and `g₋` onto the **bottom-`kdir`** (coefficient `⟨g,uᵢ⟩`, *no* `σ/η` weighting — the gradient lives in the moderate-curvature directions the old `σ·η` form suppressed), then `α₊/α₋` line-searches + a `(β,s)` grid. The **main run is pure curvature** (no gradient info). 6 panels (loss · loss-Hessian eig · function-Hessian eig · residuals · held-out test loss · ‖update‖₂) × {pos, neg, mean, best} look-aheads, plus **5 dotted baselines** — A random dirs · B shuffled ± sets · C frozen-random · D frozen-`H̄`-eigvec · **E the plain GD step**. `kdir` (the *§16/§17 kdir* control, default 32) sets the eigvectors per side — projecting onto the curvature extremes captures only a fraction of the gradient, so a larger `kdir` (more directions) descends better; the Lanczos subspace scales as `4·kdir+32` so the eigvectors are actually resolved on the clustered tanh-MLP spectrum | MSE · small `M = N·d_out` (≤ 256) |
| **17** | **per-sample** variant of §16 — instead of the averaged `H̄`, each sample projects `rₖ∇fₖ` onto the top-/bottom-`kdir` eigvectors of its **own** function Hessian `Qₖ=∇²fₖ` (`r>0` → top, `r<0` → bottom); same 6 panels + 5 baselines (incl. E·gd) | MSE · small `M = N·d_out` (≤ 256 GPU/offline, ≤ 64 in-browser) |
| **18** | the **§12b panels 1 & 2 plotted per sample** — the product `|⟨Jᵢ,u⟩|·σ·rᵢ` and alignment `|⟨Jᵢ,u⟩|/‖Jᵢ‖` onto the top-2/bottom-2 eigvectors of each `Qᵢ`, shown for **sample 1 vs sample 2** separately (instead of mean/std over samples). Reuses §12's Lanczos | **exactly 2 samples** (`nsamp = 2`) |
| **19** | correlation of two events: **A** = the one-step change in gradient norm `‖J_{t+1}ᵀr_{t+1}‖ − ‖J_tᵀr_t‖` (first-order GD prediction, `r_{t+1}=(I−(η/N)NTK)r_t`, `J_{t+1}=J_t+(η/N)Q_t(Jᵀr)`, using the run's actual `η/N` step) vs **B** = `D²(tr NTK)` (centered 2nd difference of `‖J‖²_F`). Plot 1: A & B over training; Plot 2: raw `A−B` and normalized `Â−B̂` — does the step where each first turns negative coincide? | MSE · small `N` · best with `eigevery = 1` |
| **20** (s28 / en28) | residual-weighted function Hessian `M_r=(1/N)∑ₖ rₖQₖ` spectral histograms **over training** (slider). Plot 1 = top-`K` ⊕ bottom-`K` eigenvalue histogram (matrix-free Lanczos, `K` from the **s28k** box, default 40); plots 2/3/4 = `λᵢ·\|⟨vᵢ,uₖ⟩\|` for the top-4 right-singular vectors `uₖ` of `J`. Eigenvalues `÷N` (per-sample scale) | any loss/optimizer · small `N` (`M = N·d_out ≤ grid3dcap`) |
| **21** (s29 / en29) | residual↔spectrum alignment, **three** panels (slider): (1) residual `r` onto the top-`K` NTK `K=JJᵀ` eigvecs (`σᵢ÷N`-weighted bars); (2) `J·r=∑ₖ rₖ∇fₖ` onto the top-`K` ⊕ bottom-`K` `M_r` eigvecs (signed `λ÷N`); (3) `J·r` onto the top-`K` Gauss–Newton `G=(1/N)JᵀJ` eigvecs `zᵢ` (= right-singular vecs of `J`, `cᵢ=σ÷N`). `K` from **s29k** (default 40) | multi-sample · small `N` |
| **22** (s30 / en30) **·** **23** (s31 / en31) | **quadratic-Taylor surrogate REPLACE mode** (distinct from the lockstep compare mode below). In REPLACE mode the **surrogate** trajectory (not real GD) drives §1 (loss/sharpness/residual), §2/§3 (function-Hessian / loss-Hessian / Gauss–Newton=NTK / residual spectra) and §20; §4–§21 and §24 **do not render** and a warning banner is shown. **§22** = frozen-`Q` at iteration **s30t**; **§23** = a fixed low-rank **random** symmetric `Q` (rank **s31r**, scale **s31scale**) from θ₀ as a null model. §22/§23 are **mutually exclusive** (frozen-Q wins) | MSE · small `N` |
| **24** (s32 / en32) | 1st/2nd-order `Δf` alignment time-series: decomposes the one-GD-step change in `f` into `A=JJᵀr` (NTK·r, first order) and `Bₖ=(η/2N)(J·r)ᵀQₖ(J·r)` (second order); plots `‖A‖`, `‖B‖`, `\|cos(A,r)\|`, `\|cos(B,r)\|` and eigenvalue-weighted projections of `A,B` onto the residual `r` and the top-4 NTK eigvecs `u₁–u₄` (NTK eigvals `÷N`) | GPU backend only · off in surrogate mode · `dataset ≠ owt` · small `N` |
| **25** (s33 / en33) | gradient-norm & `d/dt(J·r)` evolution; a single **linear**-axis time-series, 5 curves: `‖∇L‖`; `‖J̇·r‖=‖M_r∇L‖` (`M_r=∑ₖ rₖQₖ`); `‖J·ṙ‖=‖JᵀJ∇L‖` (the two product-rule terms of `d/dt(J·r)` under gradient flow `θ̇=−∇L`); `\|⟨∇‖J‖²_F,−Q∇L⟩\|` and `\|⟨∇‖J‖²_F,G∇L⟩\|`, where `∇‖J‖²_F=2∑ₖ Qₖ Jₖ`, and `Q=(1/N)∑ₖ Qₖ`, `G=(1/N)JᵀJ` are **both `÷N`** so the two inner products are comparable | GPU backend only · off in surrogate mode · small `N` |

A checkbox per section toggles its computation. §7a/§7/§8/§4b/§4c use the *function* NTK `Jᵤ·Jᵤᵀ` and the
generic residual `r = −∂L/∂z` (MSE: `y−f`, CE: `softmax(z)−onehot`), so they work for **cross-entropy** too;
they only need the explicit `M×p` Jacobian (`M = N·d_out`) to be feasible — skipped (stamped *"too large"*)
for e.g. OpenWebText, and empty for a single sample. **§4d** (its sign-groups need a scalar residual), **§9**
(the squared-loss σ₁ recursion) and the **§16 / §17** optimizers are MSE-only. Any §1–§10 panel a run can't
fill is stamped *"n/a"*; the §11–§17 sections stream their own records, so when one is **enabled but gated off**
(`M = N·d_out` over the §16/§17 ≤256 cap — ≤64 for §17 in-browser, `N` over the §11/§12 cap, single-sample, or the per-sample
batched Lanczos hits a memory limit) the section instead shows a clear *"§X not produced — … lower the number
of samples"* note in place of a blank frame. The per-sample sections (§12–§17) are heaviest for **multi-class**
runs, where `M = N·d_out` is large; the **2-class scalar** datasets and the small-`N` presets keep them feasible.

The cheap core (loss/sharpness/eigenvalues/§9 theory/§7a) is computed every diagnostic tick; the heaviest
slowly-varying panels — §7's FH-eigenvector projections and §8 — are computed every **`heavy every`** ticks
(default 4) and §5 SLQ ~50×/run, so a run stays responsive even with everything on. Set `heavy every` to 1 to
compute every panel every step.

## Datasets · architectures · loss · init

Set from the dropdowns (real-data / conv / transformer runs use the GPU backend):

- **Datasets** — `synthetic` (in-browser MLP), **CIFAR-10** (10-class one-hot) and **CIFAR-2**
  (2 classes `{c2a, c2b}` mapped to a single **scalar** target ±1), **MNIST** (10-class) and **MNIST-2**
  (2-class scalar ±1) — MNIST digits are zero-padded 28→32 and replicated to 3 channels, so they're a
  drop-in for the same MLP/CNN/VGG as CIFAR; **sorting** (sort a sequence; MSE), **OpenWebText**
  (GPT-2-BPE next-token LM; cross-entropy *or* MSE, minibatched — under MSE the logits are regressed
  toward the one-hot next token, normalised per token), and **Chebyshev** (Cohen et al. EoS toy task:
  `nsamp` points evenly spaced on [-1,1] labeled by the Chebyshev polynomial `T_degree`; MSE on a
  6-hidden-layer width-50 tanh MLP, ≈12.9k params. `degree` is set in the hyperparameter panel; the
  preset sharpens to the `2/η` edge of stability), and **k-sparse parity** (`ksparse`): the input is an
  `in_dim`-bit ±1 vector and the **scalar ±1 label is the parity (product) of a fixed random size-`k`
  subset** of the bits — `k` (the *k-sparse k* control) and the bit-count (`in_dim`) are set in the
  panel. It runs **in-browser with the MLP** and on the **GPU backend with the transformer** (the
  mini-GPT reads the bits as a length-`in_dim` sequence and mean-pools its per-position head to the one
  scalar parity); MSE throughout. **angle pair** (`anglepair`): a **2-sample** dataset (by default) for
  studying the geometry between two points — sample 1 is an iid-Gaussian direction scaled to `‖x₁‖ = norm1`,
  and sample 2 has `‖x₂‖ = norm2` at a **controllable angle** from x₁ (`⟨x₁,x₂⟩ = norm1·norm2·cos(angle)`).
  The `angle`, `norm1`, `norm2` and the per-sample **±1 labels** (`label x₁` / `label x₂`) are panel
  controls; `nsamp > 2` adds fresh iid-Gaussian samples at `norm1` with random ±1 labels. Residual-sign
  works (off ⇒ the ±1 labels; pos/neg ⇒ pins the residual signs). Runs in-browser with the MLP; pairs
  naturally with **§18** (per-sample projections) and **§19** (grad-norm vs `D²(NTK)`). **saddle**
  (`saddle`): a **saddle-to-saddle / staircase** task — multi-output **linear regression** on a tanh MLP
  with a diagonal teacher whose singular values are well-separated (`saddle_sep` control, σ_j = sep^j), so
  the loss descends a staircase of saddle escapes; runs **in-browser**. **agf** (`agf`): **alternating
  gradient flows** (Kunin et al. 2025, [arXiv:2506.06489](https://arxiv.org/abs/2506.06489)) — a **diagonal
  linear network** (arch `diaglin`, β = u⊙v) on a scalar `y = β*·x` with teacher coordinates `β*ᵢ = ratio^i`
  (`agf_ratio` control) and a small init, so the coordinates activate one at a time → a clean multi-jump
  staircase; GPU backend. **const** (`const`): a **constant target** — iid Gaussian `x` mapped to a fixed
  `+y` (optional `cvar` Gaussian noise), so the initial residuals naturally share a sign. The **2-class scalar**
  variants (`cifar2` / `mnist2`)
  exist so the per-sample §16/§17 optimizers run on real images: with `d_out = 1`, `M = N·1` stays small.
  Picking a dataset auto-applies its **default preset** (good `lr`/`init`/`N` + the sections that fit
  that size); the `cifar2_mlp` / `mnist_mlp` / `mnist2_mlp` presets use `lr = 0.02` (the default `0.36`
  diverges on the 3072-dim inputs).
- **Architectures** — `MLP`, `CNN`, **VGG11** (`chmul` scales channels, up to ~9.4M params),
  **mini-GPT** (sorting regressor, or a tied-embedding token-LM for OpenWebText), and **`diaglin`**
  (a **diagonal linear network** β = u⊙v, used by the `agf` alternating-gradient-flows dataset).
- **Loss** — `MSE` (all panels) or `cross-entropy` (keeps §4/§6; residual-NTK & theory panels n/a).
- **Init** — `default` (`scale/√fan_in`), `muP`, `Xavier normal`, `Xavier uniform`, or `custom` (Gaussian
  std); `init_scale` is the scale/gain/std for the chosen scheme.

Below the main widget, the **lockstep compare** mode (`mode = surrogate`) trains the real model and a
frozen-window Taylor surrogate (Eq. 1, with/without the curvature term) **side by side** and overlays their
loss, residuals, eigenvalues, the §9 / §9b / §9c / §9d σ₁-prediction panels (same Eq. 13/21/22/23/29 columns,
with the PSD term and the full-Hessian-sharpness comparison), and projections — **actual solid, surrogate
dashed**. They default
to the **EoS-demo** preset (a small tanh MLP whose loss visibly trains to the edge of stability), so the
first loss panel shows real evolution: the quadratic surrogate tracks the actual within each window, and
the linear (NTK) surrogate diverges once the frozen sharpness passes `2/η` — expected at EoS. (The wide
`width=100` presets keep the loss near-flat; their dynamics live in the sharpness panel, not the loss.)

This lockstep compare mode is **distinct** from the **§22 / §23 REPLACE mode** (above): there the surrogate
trajectory *replaces* the real run and drives the core EoS plots themselves (§1, §2/§3, §20), with §4–§21 / §24
turned off and a warning banner shown — not a side-by-side overlay.

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

## Batch runs offline (SLURM / GPU), analyse in the browser later

Run the **full computation headless** (no live server, no tab open), save **every section's records** to a
file, then **load it back into the widget** to scrub all the plots offline — so you can fan many runs out
across SLURM and your GPUs and analyse them whenever. See **[capture_README.md](capture_README.md)**.

```bash
# one capture (toggles s1–s27 plus §20/§21 (s28/s29) default ON; the §16/§17 baselines
# (s24base/s25base) and §22–§25 (s30–s33) default OFF — enable per-run via --set sNN=1):
python capture_run.py --dataset cifar10 --arch vgg11 --nsamp 25 --steps 400 \
                      --device cuda:0 --out runs_captured/cifar_vgg.json

# k-sparse parity with the transformer (or --arch mlp), e.g. 12 bits, parity of a 4-subset:
python capture_run.py --dataset ksparse --arch gpt --indim 12 --set ksparse=4 \
                      --nsamp 24 --steps 800 --device cuda:0 --out runs_captured/ksparse_gpt.json

# on SLURM (one job → one capture file); sweep_capture.sh submits a grid:
sbatch --export=ALL,ARGS="--dataset cifar10 --arch mlp --nsamp 25 --steps 400 \
                          --out runs_captured/cifar_mlp.json" run_capture.sh
```

Each run writes **one self-contained `runs_captured/*.json`**. Copy them anywhere (they need no server),
then in the widget — local `python3 -m http.server` page **or** a live backend — set **Compute → "⬆ load
run"**, pick the `.json`, and it restores that run's controls and replays every section into the plots
(§1–§9 time-series, §11–§15 cubes/grids with their sliders, §16/§17 optimizer panels). **"⬇ save run"**
downloads the most recent **GPU-backend** run in the same format (browser-compute runs aren't record-based).
This is how you "pull everything into the browser later": capture on SLURM/GPU now → load the JSONs whenever.

## Layout

```
index.html        browser widget (also served by server.py)
server.py         GPU backend — HTTP + SSE, PyTorch
capture_run.py    headless capture runner (→ a .json the widget can reload offline)
run_capture.sh    SLURM wrapper for capture_run.py · sweep_capture.sh submits a grid · capture_README.md
eos_lab/          headless package: config (presets) · models · data · train · diagnostics · plots · cli · slurm/
data/             cifar-10-batches-py/ and owt/   (git-ignored; regenerable)
runs_captured/    saved captures (git-ignored)
```

Run outputs and datasets are git-ignored — rebuild the OWT shard with `python -m eos_lab.owt_prepare` and
point `--cifar-dir` at the raw CIFAR `data_batch_*` files.
