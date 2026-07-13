# The prediction widget — how well does the theory forecast training?

This is a focused view of the main Edge-of-Stability tool. Instead of only showing what the network does, it
puts a **theory's forecast** next to the **actual** measured quantity for a series of things, so you can see
how good each forecast is.

It runs on the same `server.py` backend (no separate server) and is what you get when you open the root URL
of the server — or the `/prediction` path.

> **Reading order:** Prediction-1 → 2 → 2b → 3 → 4 → 5.1 → 5.2.
>
> **About the "evolving" forecasts (3 and 4):** each is seeded from a single snapshot and then runs forward
> on its own. Past the edge of stability the underlying update is expansive, so the forecast curve eventually
> **diverges** from the real run — the plots cap the drawn value at **20× the actual** so a runaway stays
> readable. A *"recalc from live every N steps"* control lets a forecast periodically re-sync to the real run
> (default 100; set `0` to freeze once at the start and never re-sync).

---

## Prediction-1, 2, 2b, 2c, 2d — a sweep over (learning-rate, init-scale)  ·  the "Run sweep" button

One click trains a **fresh** small network for each random `(lr, init-scale)` pair and fills four scatter
plots. Every point is one pair.

| Plot | x-axis | y-axis | "perfect" looks like… |
|------|--------|--------|-----------------------|
| **quad · t\*** | predicted sign-flip step `t*` (**quadratic** forecast) | actual `t*` | points on the `y = x` line |
| **cubic · t\*** | predicted `t*` (**cubic** forecast) | actual `t*` | points on `y = x` |
| **Σd vs Σd²L/dt²** | Σ`d` over the first N steps (signed log) | Σ of the loss's 2nd time-derivative (signed log) | a tight diagonal cloud |
| **Σd vs Σ∇‖J‖** | Σ`d` over the first N₂ steps (signed log) | Σ change in `‖J‖` (signed log) | a tight diagonal cloud |
| **2c · align vs Δ‖J‖** | `A₊−A₋` (curvature-sign-split alignment of `g=Jᵀr` with `M_r=Σ_k r_kQ_k` eigendirections; gauge-invariant) | `‖J_l‖−‖J₀‖` over `l` GD steps | a tight diagonal cloud |
| **2d · normalized** | `(A₊−A₋)/(A₊+A₋)` ∈ [−1, 1] | `(‖J_l‖−‖J₀‖)/(‖J₀‖+ε)` | a tight diagonal cloud |

Here `d = ‖J·ṙ‖ − ‖J̇·r‖` is the quantity whose **first sign-flip** marks `t*`. The predicted `t*` is computed
purely from the network state at `T_start` — nothing after `T_start` is read from the run, so the forecast is
genuinely self-contained. The two `quad · t*` / `cubic · t*` plots let you see what the extra cubic term buys.
In the two correlation plots, each point is coloured by how strongly that run's early residual lines up with
its top-3 NTK directions (shown on hover), and a rank-correlation ρ is printed in the title.

**Controls:** number of pairs, steps per pair (kept small and separate from the main run so the sweep stays
fast), K (how often the quadratic model is re-frozen), T_start, and the two sum-windows N / N₂. The sweep
runs in parallel with the main run.

**Prediction-2c / 2d** run on their own **init-scale σ-sweep** (σ log-spaced 0.1→10, one point per σ,
decoupled from the (lr, σ) pairs). With `g = Jᵀr` the residual-contracted gradient and `(σ_i, u_i)` the
top-`k` ⊕ bottom-`k` eigenpairs of `M_r = Σ_k r_kQ_k` at `t = 0`, define
`A_± = Σ_{σ_i ≷ 0} |σ_i·(u_iᵀg)| / ‖g‖` — how much `g` aligns with the **positive-** vs **negative-curvature**
directions of `M_r`. The **abs is inside** the sum, so the x-axis is **gauge-invariant** (independent of the
eigensolver's arbitrary eigenvector signs). The y-axis is the short-horizon change in the Jacobian norm,
`Δ‖J‖_F` over `l` GD steps from that init. **2c:** `x = A₊−A₋`, `y = ‖J_l‖−‖J₀‖`. **2d:** the normalized
version, `x = (A₊−A₋)/(A₊+A₋) ∈ [−1,1]`, `y = (‖J_l‖−‖J₀‖)/(‖J₀‖+ε)`. A diagonal cloud ⇒ the init
curvature-alignment predicts near-term sharpening/flattening of `‖J‖`. Controls: `l` (the *2c/2d l* box,
default 4), `k` (the *2c/2d M_r ±k* box, default 10), and the σ range. **Off by default** — flip the *show
2c/2d plots* toggle to compute + reveal them.

---

## Prediction-3 — the residual after the sign-flip  ·  cyan box  ·  *fixed J, evolving r*

At the first `t*` where `d` flips sign, freeze the Jacobian `J* = J(t*)` and run three theories for how the
error `‖r‖` shrinks afterwards, each compared to the **actual** error:

- **1st-order (frozen J):** `r_{t+1} = (I − (η/N) J*J*ᵀ) r_t`
- **2nd-order:** the same, plus a curvature correction `½(η/N)² (J*ᵀr)ᵀ Q* (J*ᵀr)` (`J*`, `Q*` frozen at `t*`)
- **evolving-J (a step function):** the Jacobian is held fixed for **k** steps (the error decays through it),
  then jumps by running the *exact* Jacobian update `Ĵ ← Ĵ + (η/N)·Q·(Ĵᵀr)` for k steps — the true
  `k`-step recurrence, not one large single step. Set the window with the **`k`** control.

**Left plot:** the two pieces of `d` and their difference (the red curve crossing 0 is `t*`). **Middle:**
`‖r‖`, actual vs the three theories (log scale). **Right:** the direction match `cos(r_actual, r_theory)`. A
second row (freeze sweep) repeats the comparison with `J*` frozen at `t*−10`, `t*+10`, `t*+20`.

The 1st- and 2nd-order curves don't depend on `k` — only the evolving-J curve changes shape with it. All
three re-sync to the live run every "recalc" window.

---

## Prediction-4 — the NTK from a freeze point  ·  teal box  ·  *fixed r, evolving J*

From a chosen iteration `t₀`, freeze the per-sample Hessian and the residual, propagate the Jacobian forward,
and read the top-3 NTK eigenvalues `λ(ĴĴᵀ)`. Two versions per eigenvalue and eigenvector:

- **frozen r (solid):** the residual-weighted Hessian `M_r = Σ_k r_k Q_k` is fixed at `t₀`, and
  `Ĵ ← Ĵ + (η/N)·M_r·Ĵ` each step.
- **evolving-Qr (dashed):** `Ĵ` still advances every step by `Ĵ ← Ĵ + (η/N)·M_r·Ĵ`, but now `M_r` is rebuilt
  from a **self-computed** residual and parameter (no live data), held fixed for **s** steps and then
  refreshed — i.e. the true `(I + (η/N) M_r)^s` power iteration rather than one coarse step. Set the window
  with the **`s`** control (`p4s`); set the freeze point with **`t₀`** (`p4t0`).

**Left:** the top-3 NTK eigenvalues (frozen · evolving · actual dots), plus the direction match `|cos|` for
the top-3 NTK eigenvectors and for the top-3 Gauss-Newton (`JᵀJ`) eigenvectors. **Second row (freeze sweep):**
the eigenvalues with `M_r` frozen at `t₀−10`, `t₀+10`, `t₀+20`.

Prediction-4 deliberately uses the **approximate** `M_r·Ĵ` update, whereas Prediction-3's evolving-J uses the
**exact** per-step Jacobian change `Q·(Ĵᵀr)`. Putting them side by side shows how good the `M_r` approximation
is. Both forecasts re-sync to the live run every "recalc" window.

---

## Prediction 5.1 — total curvature (trace)  ·  purple box

Four forecasts of `Tr(NTK)` — {quadratic, cubic} × {live residual, self residual}, each with and without a
positive-definite correction term — drawn against the **actual** `Tr(∇²L)` (a 24-probe Hutchinson estimate of
the exact loss-Hessian trace) and `Tr(JᵀJ)`. Each forecast re-anchors at the real state every window
(separate windows for the quadratic and cubic models, set in the config bar).

## Prediction 5.2 — peak curvature (sharpness)

Four forecasts of the sharpness `σ₁ = λmax(∇²L)` — {quadratic, cubic} × {live, self residual} — overlaid on
the two "actual" edges: `λmax(∇²L)` (red dashed) and `λmax(NTK)` (orange dotted). The **self**-residual
forecasts need no live residual at all (fully closed); the gap between them and the red dashed curve is the
price of that closure.

**`stat_init`** (a control): the 5.1 / 5.2 forecasts start only *after* this iteration (`0` = from the very
start); the actual curves are always shown. Handy for skipping the noisy first steps.

---

## SLQ eigen-spectrum panels (scree) + scrubber

The SLQ (Stochastic Lanczos Quadrature) panels show the **sorted eigenvalue spectrum** — a **scree plot**:
x = eigenvalue **rank** (0 = largest … `p` = smallest), y = the (signed) eigenvalue — for the function
Hessian `H`, Gauss–Newton `G`, residual term `S`, and the full loss Hessian `∇²L`. The **§4** panels have
their **own scrubber** (the *SLQ tick* slider) that rewinds through every SLQ snapshot so you can watch the
eigenspectrum evolve from the start of training, independently of the master ⏱ time cursor; the §5 copy
tracks the latest tick live.

## Also in this widget

- **Multiclass variant** — `index_prediction_multiclass.html` mirrors these forecasts for **multi-class /
  cross-entropy** output (defaults to the `maxfind` dataset with CE loss; `modadd`, CIFAR-10 and MNIST also
  available), using the p×p Gauss–Newton **Fisher** curvature. Open it at `/prediction_multiclass`.
- **Offline capture / replay** — `eos_prediction.py` (scalar) and `eos_prediction_multiclass.py` (multiclass)
  run this widget's compute **headless** and merge the prediction stream **and** the Prediction-1&2 sweep into
  one self-contained `.json` you reload with **⬆ load run** — no live server needed. See
  **[../capture_README.md](../capture_README.md)**.
- **Load ≤ iter N** — a control next to the load buttons replays only the captured records up to training
  iteration `N` (blank/0 = the whole run), handy for stepping a capture up to a chosen point.

## Where it runs

Served by the main `server.py` at the root URL and at `/prediction`:

```
http://localhost:<port>/             # e.g. http://localhost:8756/
http://localhost:<port>/prediction   # the same page
```

There's no separate launch — just start `server.py` (see the top-level README). The page turns on exactly the
computations the forecasts need and hides the other detail sections so it stays focused on the forecasts.

## Where each forecast is computed

Every forecast is computed on the server (`server.py`) and streamed to the page:

| Forecast | server.py record |
|----------|------------------|
| Prediction-3 (+ freeze sweep) | `g_pred3` / `g_pred3m` |
| Prediction-4 (+ freeze sweep) | `g_pred4` / `g_pred4m` |
| Prediction 5.1 (trace) | `g_trace` |
| Prediction 5.2 (sharpness) | §9d / §10 records |
| Predictions 1 / 2 / 2b | `run_sweep` (`sweeppt`) |
| Predictions 2c / 2d | `run_sweep` (`sweeppt2c`) |

A matching offline reference (exact autograd; agrees with the server to about `1e-3` in float64) lives in
**`eos_lab/prediction.py`**.
