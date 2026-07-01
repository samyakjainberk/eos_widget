# eos_widget_prediction — EoS prediction demo

A focused view of the main EoS widget that demonstrates the **predictive power** of the
sharpness/σ₁ theory. It reuses the same `server.py` backend (no separate server) and shows:

- **§1** (loss, sharpness λmax(∇²L) vs 2/η, residual, function-Hessian edges),
- **§2 / §3** (top-n / bottom-n eigenvalue evolution of H / ∇²L / GN / S),
- **§21 panels 1 & 2** (residual ↔ NTK-spectrum, and J·r ↔ M_r-spectrum alignment),
- a **prediction panel** (4 plots) forecasting σ₁ against the *actual* NTK edge λmax(JJᵀ)
  **and** the loss-Hessian sharpness λmax(∇²L):
  1. frozen-Q first-order prediction (Eq-21) driven by the **live** residual (= §9c, 2nd plot, + the NTK line),
  2. the same Eq-21 prediction with the residual **self-computed** from the frozen quadratic model (= §9d-c, 2nd plot, + the NTK line),
  3. the **cubic** prediction (Eq-51) driven by the **live** residual (= §10, 2nd plot, + the ∇²L line),
  4. the cubic prediction with the residual **self-computed** from the quadratic model — a **new**
     `cubic_self_step` computation in `server.py` that fuses §10's cubic J/Q/T propagation with §9d's
     r_q self-trajectory. Shown vs both λmax(NTK) and λmax(∇²L).

`c51d`/`c51dp` are **server-computed** (like the §22–§26 prediction quantities): the demo defaults to the
GPU/server backend, and the browser only renders them. (An `eos_lab` port for offline parity is a
straightforward follow-up — the live-residual `c51` reference is already in `eos_lab._cubic_step`.)

Plots 2 & 4 are the "fully-closed" forecasts: they need **no live-run residual**, so the gap to the
dashed actual curves is the price of that closure.

## Serving

Served by the main `server.py` at the route **`/prediction`** (the page's `/run` requests are
root-absolute, so use the no-trailing-slash path). On the reverse-tunnelled SLURM fleet it is reachable at:

```
http://localhost:<PORT>/prediction        # e.g. http://localhost:8756/prediction
```

No separate launch script is needed — restart the existing fleet (`run_serve_localhost.sh`) with the
current `server.py` and the route is live. The page forces `s1,s2,s3,s12,s14,s16,s17,s29=1` in
`currentQueryParams()` so the §9/§9c/§9d-c/§10/§21 quantities are always computed (their multi-column
detail sections are CSS-hidden to keep the demo focused on the forecast panel).

## New backend quantity

`cubic_self_step` (server.py) emits `c51d` / `c51dp`: the Eq-51 cubic σ₁ prediction driven by
r_q = Y − (f₀ + J₀Δθ + ½ΔθᵀQ₀Δθ) instead of the live residual, with Δθ advanced by the same self-gradient.
At each cubic window start Δθ=0 ⇒ r_q = live residual ⇒ `c51d` = `c51` = the actual NTK edge (verified invariant).
