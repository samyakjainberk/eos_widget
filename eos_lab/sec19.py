"""§19 (s27) — one-step-ahead change in GRADIENT NORM A_t + tr(NTK), MIRRORS server._sec19_payload
(server.py:1684-1704) / index.html runLocal.

The quantity is the first-order (one GD step ahead) change in the gradient norm under the run's ACTUAL GD
step (effective rate η_eff = lr/N, because the loss is the MEAN over the N input samples):

    A_t = ‖J_{t+1}ᵀ r_{t+1}‖ − ‖J_tᵀ r_t‖

with the first-order GD predictions (MSE residual, output-Hessian A = I):
    r_{t+1} = (I − η_eff · NTK_t) r_t ,   NTK = J Jᵀ  (M×M)
    J_{t+1} = J_t + η_eff · Q_t (J_tᵀ r_t) ,   Q_t (·) = Σ_k r_{t+1,k} ∇²f_k (·)   (the residual-weighted function Hessian).
Writing g = J_tᵀ r_t (the user's "J r"), and Jg ≡ J_t (M×p):
    J_{t+1}ᵀ r_{t+1} = g − η_eff · Jᵀ(J g) + η_eff · (Σ_k r_{t+1,k} Q_k) g
the last term via hvpS (residual-weighted function-Hessian HVP). tr(NTK) = ‖J‖²_F = Σ Jg² is also returned
(the browser forms the centered 2nd-difference B_t = D²(trNTK) client-side from the time series).

Cheap: a few matvecs + ONE hvpS (= eos_lab hvp_S, exact autograd). The eos_lab multi-sample primitives
jac_cols / hvp_S mirror server's FD definitions, so the record matches server 1e-5..1e-13 in fp64.

GATE (server.py:3843): §19 runs only for MSE + GD (the formula hardcodes the MSE residual update
r_{t+1} = (I − η/N·NTK) r, i.e. output-Hessian A = I; CE needs A = diag(p) − ppᵀ ≠ I) and only when
(N·outD) ≤ grid3dcap, with Jc / rr available. The caller (diagnostics.py at integration time) enforces this.
"""
import torch

from .models import jac_cols, hvp_S


def sec19_payload(model, Jc, rr, th, X, lr, N, outD):
    """§19 record from the per-output Jacobian Jc (M,p) and generic residual rr (M,) — the SAME inputs
    diagnostics.compute() already has in scope (Jc = jac_cols(...)[0], rr = (Y−f).reshape(-1)).

    Mirrors server._sec19_payload(Jc, rr, th, X, lr, N, outD) line-by-line. Note the eos_lab hvp_S signature
    is hvp_S(model, theta, X, c, v) — c (cotangent) BEFORE v — so server's hvpS(th, X, g, r_next.reshape(N,outD))
    maps to hvp_S(model, th, X, r_next.reshape(N, outD), g).

    Returns {"A", "trNTK", "gn", "gn_next"}.
    """
    M = N * outD
    Jg = Jc[:M]                                              # M×p effective-sample Jacobian
    r = rr[:M]                                               # M residual (Y − f)
    g = Jg.t() @ r                                           # p   gradient g = Jᵀr  ("J r")
    gn = float(g.norm())                                    # ‖J_t r_t‖
    etaN = lr / max(N, 1)                                    # actual GD step (mean loss ⇒ η/N)
    Jgg = Jg @ g                                             # M   NTK·r = J Jᵀ r
    r_next = r - etaN * Jgg                                  # (I − η_eff·NTK) r
    JtJg = Jg.t() @ Jgg                                      # p   Jᵀ J g
    Hg = hvp_S(model, th, X, r_next.reshape(N, outD), g)    # (Σ_k r_{t+1,k} Q_k) g  (residual-weighted func-Hessian HVP)
    g_next = g - etaN * JtJg + etaN * Hg                     # J_{t+1}ᵀ r_{t+1}
    gn_next = float(g_next.norm())                          # ‖J_{t+1} r_{t+1}‖
    return {"A": gn_next - gn, "trNTK": float((Jg * Jg).sum()), "gn": gn, "gn_next": gn_next}


def sec19_from_theta(model, loss, th, X, Y, lr, N, outD):
    """Convenience standalone entry: build Jc / rr at θ (exactly as diagnostics.compute would) and return the
    §19 record. Generic residual rr = Y − f matches server's (−N·resid_cotangent).reshape(-1) for MSE.
    Used for standalone parity verification; the integration path calls sec19_payload directly with the
    Jc / rr already in scope."""
    Jc, _out_flat = jac_cols(model, th, X)                  # _out_flat is (M,) — DON'T feed it to resid_cotangent
    out = model.forward(th, X)                              # (N, outD) — resid_cotangent needs the unflattened shape
    rr = (-N * loss.resid_cotangent(out, Y, N)).reshape(-1)
    return sec19_payload(model, Jc, rr, th, X, lr, N, outD)
