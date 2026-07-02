# eos_widget_prediction — Progressive-Sharpening prediction demo

A focused view of the main EoS widget that tests **how well the Progressive-Sharpening / Edge-of-Stability
theory predicts training dynamics**. It reuses the main `server.py` backend (no separate server) at the
route **`/prediction`** and is a sequence of numbered forecasts, each comparing a **theoretical
prediction** to the **actual** measured quantity over training.

> Panel order (intentional): **Prediction-1 → 2 → 2b → 3 → 4 → 5.1 → 5.2**

---

## The predictions

### Prediction-1, 2, 2b — sweep over (learning-rate, init-std)  · `Run sweep`
One click trains a *fresh* small net for every random `(lr, init_std)` pair and plots three scatters:

| Panel | x-axis | y-axis |
|-------|--------|--------|
| **Prediction-1** | predicted sign-change iteration `t*` | actual `t*` — first step where `d = ‖J·ṙ‖ − ‖J̇·r‖` changes sign |
| **Prediction-2** | `sign(d)` at `t=0` | `sign` of the peak `d²loss/dt²`  (Pearson r shown) |
| **Prediction-2b** | `sign(d)` at `t=0` | `sign` of the **‖J‖ trend** at the start of training (rising +1 / falling −1) |

The predicted `t*` (Prediction-1) is a **cubic** forecast from `θ(T_start)` — nothing after `T_start` is read
from the run. Controls: **pairs**, **steps/pair** (`swsteps` — kept small and *separate* from the main run's
step count so the sweep stays fast), **K** (quad refreeze), **T_start**, **sweep on port** (offload the heavy
sweep to a second fleet GPU; auto-falls-back to this server if that port is unreachable).

### Prediction-3 — linear + quadratic residual theory after the sign-change  · cyan box, 2 panels
At the first `t*` where `d = ‖J·ṙ‖ − ‖J̇·r‖` changes sign, freeze `J*=J(t*)` and `Q*=Q(t*)` and evolve the
frozen-`J` NTK residual theory, comparing `‖r‖` and direction to the **actual** residual thereafter:

- **1st-order:** `r_{t+1} = (I − (η/N) J*J*ᵀ) r_t`
- **2nd-order:** `+ ½(η/N)² (J*ᵀr)ᵀ Q* (J*ᵀr)`   ( `Q*`, `J*` both frozen at `t*` )

**Panel 1:** the two norms + their difference (sign-change = red crosses 0), `‖r‖` actual vs 1st/2nd-order
theory (log), and `cos(r_actual, r_theory)`.  **Panel 2 (freeze sweep):** the same, freezing `J*,Q*` at
`t*−10`, `t*+10`, `t*+20`.

### Prediction-4 — frozen- vs evolving-residual NTK propagation  · teal box, 2 panels
From iteration `t₀` freeze the function-Hessian `Q(θ₀)` and propagate `Ĵ_{s+1} = (I + (η/N) M_r) Ĵ_s`,
reading the top-3 NTK eigenvalues `λ(ĴĴᵀ)`. Two residual models per eigenvalue/eigenvector:

- **frozen r** (solid): `M_r = Σ_k r_k Q_k` fixed at `t₀`
- **evolving r** (dashed): the residual evolves via `r_{t+1} = (I − (η/N) ĴĴᵀ) r_t`, **re-anchored** to the
  actual residual every **s** steps (`p4s`); `M_r` recomputed each step

**Panel 1:** top-3 NTK eigenvalues (frozen · evolving · actual dots) + `|cos|` of actual-vs-predicted top-3
**NTK** eigenvectors + `|cos|` of top-3 **Gauss-Newton** (`JᵀJ`) eigenvectors.  **Panel 2 (freeze sweep):**
top-3 NTK eigenvalues with `M_r` frozen at `t₀−10`, `t₀+10`, `t₀+20`.  Controls: `t₀` (`p4t0`), re-anchor `s`
(`p4s`).

### Prediction 5.1 (Trace) — predicted Tr(NTK) vs Tr(∇²L) & Tr(JᵀJ)  · purple box
Four forecasts of `Tr(NTK)` — {quadratic, cubic} × {live residual, self residual}, each ±PSD term — vs the
**actual** `Tr(∇²L)` (24-probe Hutchinson of the exact loss-Hessian) and `Tr(JᵀJ)`. Each trajectory
**re-anchors** at the actual state every window: quadratic every `every_iter_quadric_approx`, cubic every
`every_iter_cubic_approx`.

### Prediction 5.2 (Sharpness) — predicted σ₁ vs λmax(∇²L) & λmax(NTK)
Four forecasts of the sharpness `σ₁ = λmax(∇²L)` — {quadratic (Eq-21), cubic (Eq-51)} × {live, self
residual} — overlaid on the two actual edges `λmax(∇²L)` (red dashed) and `λmax(NTK)` (orange dotted). The
**self**-residual plots need no live residual (the fully-closed forecast); their gap to the dashed curve is
the price of that closure.

### `stat_init` — when the forecasts begin
The **5.1 & 5.2** theoretical forecasts begin only *after* iteration `stat_init` (`0` = from the start); the
actual curves are shown throughout. Handy for skipping the noisy first steps.

---

## Serving
Served by the main `server.py` at **`/prediction`** (its `/run` requests are root-absolute → use the
no-trailing-slash path). On the reverse-tunnelled SLURM fleet:

```
http://localhost:<PORT>/prediction        # e.g. http://localhost:8756/prediction
```

No separate launch script — restart the fleet (`run_serve_localhost.sh`) with the current `server.py`. The
page forces `s1,s2,s3,s12,s14,s16,s17,s29,s36,s37,s38 = 1` in `currentQueryParams()` so all the
§9/§9c/§9d-c/§10/§21 + prediction quantities are always computed (their detail sections are CSS-hidden to
keep the demo focused on the forecasts).

## Backend
The forecasts are **server-computed** (`server.py`):

| Forecast | server.py records |
|----------|-------------------|
| Prediction-3 / 3 freeze | `g_pred3` / `g_pred3m` |
| Prediction-4 / 4 freeze | `g_pred4` / `g_pred4m` |
| Prediction 5.1 (Trace)  | `g_trace` |
| Prediction 5.2 (Sharpness) | §9d/§10 `thP` / `thP_d` / `c51` / `c51d` records |
| Predictions 1 / 2 / 2b  | `run_sweep` |

A faithful offline reference port lives in **`eos_lab/prediction.py`** (exact-autograd; matches the FD
server backend to ~1e-3 in float64).
```