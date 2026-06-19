"""
Static matplotlib panels — the headless counterpart of the webpage's Plotly rows.

MIRRORS:  index.html plot rows  ↔  the §1/§2/§3/§5/§6 records produced by diagnostics.py.
Each function renders one section's row; `save_panels()` writes every supported row to PNGs.
"""
import math
import os

import matplotlib
matplotlib.use("Agg")          # headless: render to files, no display needed
import matplotlib.pyplot as plt


def _finite(xs, ys):
    return [x for x, y in zip(xs, ys) if y == y], [y for y in ys if y == y]


def plot_section1(series, meta, ax=None):
    """§1 — loss (train+test), sharpness vs 2/η, residual rms, spectral edges of H."""
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.4)) if ax is None else (ax.figure, ax)
    t = series["t"]
    axs[0].plot(t, series["loss"], label="train")
    if "test_loss" in series:
        axs[0].plot(*_finite(t, series["test_loss"]), ls="--", label="test")
        axs[0].legend()
    axs[0].set_yscale("log"); axs[0].set_title("loss"); axs[0].set_xlabel("step")
    if "sharpness" in series:
        axs[1].plot(t, series["sharpness"], label="sharpness")
        axs[1].axhline(meta["thr"], ls="--", c="r", label="2/η (EoS)")
        axs[1].set_title("sharpness vs edge of stability"); axs[1].legend(); axs[1].set_xlabel("step")
    if "resid_head" in series:              # per-sample residual y−f for ALL samples (nResid=M) — matches the widget's panel
        for j, track in enumerate(series["resid_head"]):
            axs[2].plot(*_finite(t, track), lw=1)
        axs[2].axhline(0, c="k", lw=0.6)
        axs[2].set_title("residual  y − f (per sample)"); axs[2].set_xlabel("step")
    elif "resid_rms" in series:             # fallback: rms summary
        axs[2].plot(t, series["resid_rms"]); axs[2].set_title("residual rms |y−f|"); axs[2].set_xlabel("step")
    else:                                   # cross-entropy: no residual r=y−f to show
        axs[2].axis("off")
    if "H_edge_max" in series:
        axs[3].plot(t, series["H_edge_max"], label="λ_max(H)")
        axs[3].plot(t, series["H_edge_min"], label="λ_min(H)")
        axs[3].axhline(0, c="k", lw=0.6); axs[3].set_title("function-Hessian spectral edges")
        axs[3].legend(); axs[3].set_xlabel("step")
    fig.tight_layout()
    return fig


def _plot_eig_row(series, which, title):
    """§2 (top) / §3 (bottom): eigenvalue tracks of H, loss Hessian, G, S."""
    mats = [("H_" + which, "function Hessian H"), ("lossH_" + which, "loss Hessian (G+S)"),
            ("G_" + which, "Gauss–Newton G"), ("S_" + which, "residual term S")]
    present = [(k, lab) for k, lab in mats if k in series]
    if not present:
        return None
    fig, axs = plt.subplots(1, len(present), figsize=(4.5 * len(present), 3.4), squeeze=False)
    axs = axs[0]
    t = series["t"]
    for ax, (k, lab) in zip(axs, present):
        for j, track in enumerate(series[k]):
            ax.plot(t, track, label=f"λ{j+1}")
        ax.axhline(0, c="k", lw=0.6); ax.set_title(f"{title} — {lab}"); ax.set_xlabel("step")
        if len(series[k]) <= 4:
            ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_section2(series):
    return _plot_eig_row(series, "top", "top-n eigenvalues")


def plot_section3(series):
    return _plot_eig_row(series, "bot", "bottom-n eigenvalues")


def plot_section6(series):
    """§6 — eigenspace rotation: max/mean principal angle of the ±-energy subspaces of H."""
    keys = [k for k in ("rot_pos_max", "rot_neg_max") if k in series]
    if not keys:
        return None
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.4))
    t = series["t"]
    if "rot_pos_max" in series:
        ax.plot(*_finite(t, series["rot_pos_max"]), label="positive subspace (max)")
        ax.plot(*_finite(t, series["rot_pos_mean"]), ls="--", label="positive (mean)")
    if "rot_neg_max" in series:
        ax.plot(*_finite(t, series["rot_neg_max"]), label="negative subspace (max)")
        ax.plot(*_finite(t, series["rot_neg_mean"]), ls="--", label="negative (mean)")
    ax.set_title("§6 eigenspace rotation of H (principal angle, °)"); ax.set_xlabel("step")
    ax.legend(fontsize=8); fig.tight_layout()
    return fig


def plot_slq(history, n_samp=1):
    """§5 — the most recent SLQ spectral densities (H, loss Hessian, G, S).
    The H density is the only one on the summed (∇²Σf) scale — the loss-Hessian/G/S densities already
    carry the 1/N from the averaged loss — so its eigenvalue axis is divided by N to match the §1–§3 H
    tracks (density rescaled by N to stay normalised). slq_H is a plot-only input, so this lives here."""
    last = next((r for r in reversed(history) if "slq_H" in r), None)
    if last is None:
        return None
    panels = [(k, lab) for k, lab in [("slq_H", "H"), ("slq_lossH", "loss Hessian"),
                                      ("slq_G", "G"), ("slq_S", "S")] if k in last]
    fig, axs = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 3.2), squeeze=False)
    for ax, (k, lab) in zip(axs[0], panels):
        x, y = last[k]["x"], last[k]["y"]
        if k == "slq_H" and n_samp > 1:
            x = [v / n_samp for v in x]; y = [v * n_samp for v in y]
        ax.plot(x, y)
        # LOG density: Hessian spectra are a huge bulk near 0 plus outlier clusters ~10³–10⁴× smaller —
        # on a linear axis only the bulk is visible (looks like one cluster). Log-y exposes the outlier
        # bulges, the standard way to read a SLQ spectral density (Ghorbani et al. / PyHessian).
        pos = [v for v in y if v > 0]
        if pos:
            ax.set_yscale("log")
            ax.set_ylim(max(min(pos), max(pos) * 1e-8), max(pos) * 1.5)
        ax.set_title(f"SLQ density — {lab}"); ax.set_xlabel("eigenvalue"); ax.set_ylabel("density (log)")
    fig.tight_layout()
    return fig


def plot_section5_trace(history):
    """§5 — trace of each operator over training (Hutchinson). One line per operator vs iteration."""
    snaps = [r for r in history if "tr_HL" in r]
    if len(snaps) < 2:
        return None
    keys = [("tr_H", "trace H  (÷N)", "#2563eb"), ("tr_G", "trace G", "#16a34a"),
            ("tr_S", "trace S", "#d35400"), ("tr_HL", "trace ∇²L (loss Hessian)", "#dc2626")]
    present = [k for k in keys if any(k[0] in r for r in snaps)]
    fig, axs = plt.subplots(1, len(present), figsize=(4.5 * len(present), 3.2), squeeze=False)
    for ax, (key, lab, col) in zip(axs[0], present):
        xs = [r["t"] for r in snaps if key in r]; ys = [r[key] for r in snaps if key in r]
        ax.plot(xs, ys, color=col); ax.axhline(0, c="k", lw=0.6)
        ax.set_title(lab, fontsize=10); ax.set_xlabel("step"); ax.set_ylabel("trace")
    fig.suptitle("§5 — trace of each operator over training (Hutchinson)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def _tracks_panel(series, groups, suptitle):
    """Generic multi-subplot panel: `groups` = [(series_key, title), ...]; each present key → one subplot
    with one line per track index. Used for the §4/§7/§8/§4b–d projection panels."""
    present = [(k, lab) for k, lab in groups if k in series]
    if not present:
        return None
    fig, axs = plt.subplots(1, len(present), figsize=(4.4 * len(present), 3.3), squeeze=False)
    t = series["t"]
    for ax, (k, lab) in zip(axs[0], present):
        for j, track in enumerate(series[k]):
            ax.plot(*_finite(t, track), label=f"{j+1}")
        ax.axhline(0, c="k", lw=0.6); ax.set_title(lab); ax.set_xlabel("step")
        if len(series[k]) <= 6:
            ax.legend(fontsize=7)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_section4(series):
    return _tracks_panel(series, [("jtN", "J·v_top(H) /‖J‖"), ("jbN", "J·v_bot(H) /‖J‖"),
                                  ("jt", "J·v_top(H) raw"), ("jb", "J·v_bot(H) raw")],
                         "§4 — J = ∇Σf onto eigenvectors of H")


def plot_section7a(series):
    """§7a — NTK alignment + the change-product diagnostics. Up to 4 subplots, matching the widget:
    (1) residual onto top-n NTK eigvecs; (2) the per-step product Δλ·Δ⟨r,vₖ⟩ (solid) with its running
    time-average (dashed); (3) the 1-step-lagged product Δλ(t)·Δ⟨r,vₖ⟩(t-1) likewise; (4) NTK u₁→FH-SVD.
    Δx(t)=x(t)-x(t-1); λ = sharpness, rₖ = residual·NTK-eigvecₖ."""
    panels = []
    if "ntkR" in series: panels.append("ntkR")
    if "ntkGs" in series: panels.append("g7s")
    if "ntkGl" in series: panels.append("g7l")
    if "ntkH" in series: panels.append("ntkH")
    if not panels:
        return None
    titles = {"ntkR": "residual·NTK eigvec", "g7s": "Δλ·Δ⟨r,vₖ⟩  (dashed = run-avg)",
              "g7l": "Δλ(t)·Δ⟨r,vₖ⟩(t-1)  (dashed = run-avg)", "ntkH": "NTK u₁·FH y_k"}
    t = series["t"]
    fig, axs = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.3), squeeze=False)
    for ax, key in zip(axs[0], panels):
        if key in ("ntkR", "ntkH"):
            for j, track in enumerate(series[key]):
                ax.plot(*_finite(t, track), label=f"{j+1}")
        else:
            prod = "ntkGs" if key == "g7s" else "ntkGl"
            avg = prod + "A"
            for j, track in enumerate(series[prod]):
                line, = ax.plot(*_finite(t, track), label=f"{j+1}")
                if avg in series and j < len(series[avg]):
                    ax.plot(*_finite(t, series[avg][j]), ls=":", lw=1.0, c=line.get_color())
        ax.axhline(0, c="k", lw=0.6); ax.set_title(titles[key]); ax.set_xlabel("step")
        ax.legend(fontsize=7)
    fig.suptitle("§7a — NTK alignment + Δsharpness·Δ(residual·NTK) products", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def _tracks_grid(series, groups, suptitle, ncol=5):
    """Like _tracks_panel but WRAPS many tracks into a grid (ncol per row) — for sections the widget
    shows as a long strip of panels (§7's 10, §4b's 8). Each present key → one subplot; unused cells hidden."""
    present = [(k, lab) for k, lab in groups if k in series]
    if not present:
        return None
    n = len(present); nrow = (n + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.2 * nrow), squeeze=False)
    t = series["t"]
    for idx, (k, lab) in enumerate(present):
        ax = axs[idx // ncol][idx % ncol]
        for j, track in enumerate(series[k]):
            ax.plot(*_finite(t, track), label=f"{j+1}")
        ax.axhline(0, c="k", lw=0.6); ax.set_title(lab, fontsize=8.5); ax.set_xlabel("step")
        if len(series[k]) <= 6:
            ax.legend(fontsize=7)
    for idx in range(n, nrow * ncol):
        axs[idx // ncol][idx % ncol].axis("off")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_section7(series):
    """§7 — all 10 widget tracks: reshaped function-Hessian (M₁,M₂) top/bottom-2 eigenvalues, and the
    left-singular vectors of J projected onto the top-2 / bottom-2 eigvecs of M₁ and M₂."""
    return _tracks_grid(series, [
        ("fhEvT", "reshaped FH M₁,M₂ — top-2 eigenvalues"),
        ("fhEvB", "reshaped FH M₁,M₂ — bottom-2 eigenvalues"),
        ("jhe1", "J left-sing onto M₁ top eigvec e₁"),
        ("jhe2", "J left-sing onto M₁ 2nd eigvec e₂"),
        ("jhe1b", "J left-sing onto M₁ bottom eigvec e₋₁"),
        ("jhe2b", "J left-sing onto M₁ 2nd-bottom eigvec e₋₂"),
        ("jh2e1", "J left-sing onto M₂ top eigvec e₁"),
        ("jh2e2", "J left-sing onto M₂ 2nd eigvec e₂"),
        ("jh2e1b", "J left-sing onto M₂ bottom eigvec e₋₁"),
        ("jh2e2b", "J left-sing onto M₂ 2nd-bottom eigvec e₋₂"),
    ], "§7 — multi-sample NTK & function-Hessian tensor SVD", ncol=5)


def plot_section8(series):
    return _tracks_panel(series, [("g2Jn", "vec(J)·G2 singvec /‖J‖_F"), ("g2J", "vec(J)·G2 singvec")],
                         "§8 — vec(J) onto FH-reshape singular vectors")


def plot_section4b(series):
    """§4b — all 8 widget tracks: J·r onto the top/bottom-n eigvecs of Q[u₁], raw and normalized by
    ‖J·r‖ and by ‖r‖, plus the Gauss–Newton top eigvec g₁ onto the same Q[u₁] eigvecs."""
    return _tracks_grid(series, [
        ("q9t", "J·r onto top-n eigvecs of Q[u₁]"),
        ("q9b", "J·r onto bottom-n eigvecs of Q[u₁]"),
        ("q9tN", "J·r/‖J·r‖ onto top-n eigvecs of Q[u₁]"),
        ("q9bN", "J·r/‖J·r‖ onto bottom-n eigvecs of Q[u₁]"),
        ("q9tR", "J·r/‖r‖ onto top-n eigvecs of Q[u₁]"),
        ("q9bR", "J·r/‖r‖ onto bottom-n eigvecs of Q[u₁]"),
        ("q9gt", "GN top eigvec g₁ onto top-n eigvecs of Q[u₁]"),
        ("q9gb", "GN top eigvec g₁ onto bottom-n eigvecs of Q[u₁]"),
    ], "§4b — J·r onto eigenvectors of Q[u₁]", ncol=4)


def plot_section4c(series):
    return _tracks_panel(series, [("q10N", "Q[u₁]Jr→GN eigvec /‖·‖"), ("q10", "Q[u₁]Jr→GN eigvec")],
                         "§4c — Q[u₁]·(J·r) onto Gauss–Newton eigenvectors")


def plot_section4d(series):
    """§4d — per-residual-sign-group projections. The + and − groups are drawn in SEPARATE panels
    (top row = +group, bottom row = −group) instead of overlaid in one, so the two groups stay
    readable (mirrors the widget's split surrogate §4d panels)."""
    groups = [("d4Ht", "J→H top"), ("d4Hb", "J→H bot"), ("d4Qt", "J·r→Q top"), ("d4Qb", "J·r→Q bot")]
    present = [(k, lab) for k, lab in groups if k in series]
    if not present:
        return None
    t = series["t"]
    ncol = len(present)
    fig, axs = plt.subplots(2, ncol, figsize=(4.4 * ncol, 6.2), squeeze=False)
    for col, (k, lab) in enumerate(present):
        tracks = series[k]
        half = max(1, len(tracks) // 2)                 # first half = +group, second half = −group
        for row, (sign, lo, hi) in enumerate([("+group", 0, half), ("−group", half, len(tracks))]):
            ax = axs[row][col]
            for j in range(lo, hi):
                ax.plot(*_finite(t, tracks[j]), label=f"{j - lo + 1}")
            ax.axhline(0, c="k", lw=0.6); ax.set_title(f"{sign} · {lab}"); ax.set_xlabel("step")
            if hi - lo <= 6:
                ax.legend(fontsize=7)
    fig.suptitle("§4d — per-residual-sign-group projections (+ / − groups split)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_section9(series, meta, pkey="thP", ppsdkey="thPpsd", show_pred=True, show_psd=True,
                  suptitle="§9 — theoretical vs empirical sharpness"):
    """§9 — theory (Eq-13/21/22/23/29) vs empirical σ₁. Only the columns with data are drawn:
    Eq-13 is single-sample (M==1) so it is empty for multi-output datasets, and the whole panel
    is skipped when neither theory nor empirical produced any finite point (e.g. multi_ok=False).
    The widget splits this into separate rows: §9 (predicted) and §9b (predicted+PSD) — show_pred/show_psd
    pick which. §9d reuses this with pkey='thP_d' (the quadratically self-computed-residual predictions)."""
    thP = series.get(pkey); thA = series.get("thA"); thPpsd = series.get(ppsdkey)
    if thP is None and thA is None:
        return None
    t = series["t"]
    labels = ["Eq-13 (single)", "Eq-21 (col-2)", "Eq-22 (col-3)", "Eq-23 (col-4)", "Eq-29 (col-5)"]

    def col(arr, i):
        return arr[i] if (arr is not None and i < len(arr)) else None

    # keep only columns with a finite point in whichever curves we're drawing (predicted / +PSD / empirical)
    present = [i for i in range(5)
               if (show_pred and any(y == y for y in (col(thP, i) or [])))
               or (show_psd and any(y == y for y in (col(thPpsd, i) or [])))
               or any(y == y for y in (col(thA, i) or []))]
    if not present:
        return None

    fig, axs = plt.subplots(1, len(present), figsize=(4.5 * len(present), 3.3), squeeze=False)
    for ax, i in zip(axs[0], present):
        emp = [y for y in (col(thA, i) or []) if y == y]
        prd = [y for y in (col(thP, i) or []) if y == y]
        prdP = [y for y in (col(thPpsd, i) or []) if y == y]
        if show_pred and col(thP, i) is not None:
            ax.plot(*_finite(t, col(thP, i)), label="predicted")
        if show_psd and col(thPpsd, i) is not None:          # §9b: prediction + 2nd-order PSD term ‖ΔJᵀu₁‖²
            ax.plot(*_finite(t, col(thPpsd, i)), label="predicted + PSD")
        if col(thA, i) is not None:
            ax.plot(*_finite(t, col(thA, i)), ls="--", label="empirical σ₁")
        # The frozen-window theory spikes at each window restart; clamp the y-range to the empirical
        # band (with headroom) so the tracking region stays readable instead of being crushed flat.
        ref = emp or prd or prdP
        if ref:
            lo, hi = min(ref), max(ref)
            pad = 0.3 * (hi - lo) + 1e-9
            ax.set_ylim(min(lo - pad, 0.0) if lo >= 0 else lo - pad, hi + pad)
        ax.set_title(labels[i]); ax.set_xlabel("step"); ax.legend(fontsize=8)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_section9c(series, meta, pkey="thP", ppsdkey="thPpsd",
                   suptitle="§9c — σ₁ predictions vs the full-Hessian sharpness"):
    """§9c — the same Eq-13/21/22/23/29 σ₁ predictions, but compared against the FULL loss-Hessian
    sharpness λmax(∇²L)=λmax(G+S) (the EoS quantity) instead of the Gauss-Newton edge λmax(G).
    Each column: predicted, predicted+PSD, and the full-Hessian sharpness; the gap is the residual S.
    §9d-c reuses this with pkey='thP_d' (the quadratically self-computed-residual predictions)."""
    thP = series.get(pkey); thPpsd = series.get(ppsdkey); thAH = series.get("thAH")
    if thP is None or thAH is None or not any(y == y for y in thAH):
        return None
    t = series["t"]
    labels = ["Eq-13 (single)", "Eq-21 (col-2)", "Eq-22 (col-3)", "Eq-23 (col-4)", "Eq-29 (col-5)"]

    def col(arr, i):
        return arr[i] if (arr is not None and i < len(arr)) else None

    present = [i for i in range(5) if any(y == y for y in (col(thP, i) or []))]
    if not present:
        return None
    fig, axs = plt.subplots(1, len(present), figsize=(4.5 * len(present), 3.3), squeeze=False)
    for ax, i in zip(axs[0], present):
        ax.plot(*_finite(t, col(thP, i)), label="predicted")
        if col(thPpsd, i) is not None:
            ax.plot(*_finite(t, col(thPpsd, i)), label="predicted + PSD")
        ax.plot(*_finite(t, thAH), ls="--", color="tab:red", label="sharpness λmax(∇²L)")
        ref = [y for y in thAH if y == y] or [y for y in (col(thP, i) or []) if y == y]
        if ref:
            lo, hi = min(ref), max(ref)
            pad = 0.3 * (hi - lo) + 1e-9
            ax.set_ylim(min(lo - pad, 0.0) if lo >= 0 else lo - pad, hi + pad)
        ax.set_title(labels[i]); ax.set_xlabel("step"); ax.legend(fontsize=8)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_section9b(series, meta):
    """§9b — the §9 predictions PLUS the dropped 2nd-order PSD term ‖ΔJᵀu₁‖² (a sharpening floor),
    vs empirical σ₁ (the widget's separate predicted+PSD row)."""
    return plot_section9(series, meta, show_pred=False, show_psd=True,
                         suptitle="§9b — predicted + PSD term ‖ΔJᵀu₁‖² vs empirical σ₁")


def plot_section9d(series, meta):
    """§9d — §9 predictions but with the residual SELF-COMPUTED by the frozen quadratic model (vs empirical σ₁)."""
    return plot_section9(series, meta, pkey="thP_d", ppsdkey="thPpsd_d", show_psd=False,
                         suptitle="§9d — predicted σ₁ (quadratic self-residual) vs empirical")


def plot_section9dc(series, meta):
    """§9d-c — the §9d (quadratic self-residual) predictions vs the full loss-Hessian sharpness."""
    return plot_section9c(series, meta, pkey="thP_d", ppsdkey="thPpsd_d",
                          suptitle="§9d-c — quad-self-residual σ₁ predictions vs the full-Hessian sharpness")


def _cubic_pred_ax(ax, t, series, key, keyP, act, title, name):
    """One §10 prediction subplot: the cubic-approximation prediction (no-PSD and +PSD) vs the actual `act`."""
    pred = series.get(key); predP = series.get(keyP)
    has = ((pred is not None and any(y == y for y in pred))
           or (predP is not None and any(y == y for y in predP)))
    if not has:                                  # Eq-47 is single-sample-only, Eq-51 multi-only → blank the other
        ax.text(0.5, 0.5, "n/a\n(other sample regime)", ha="center", va="center",
                fontsize=9, color="#94a3b8")
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        return
    if pred is not None:
        ax.plot(*_finite(t, pred), label=name)
    if predP is not None:
        ax.plot(*_finite(t, predP), label=name + " + PSD")
    ax.plot(*_finite(t, act), ls="--", color="tab:red", label="empirical")
    ref = [y for y in act if y == y]             # clamp y to the empirical band (frozen-window spikes stay off-screen)
    if ref:
        lo, hi = min(ref), max(ref); pad = 0.3 * (hi - lo) + 1e-9
        ax.set_ylim(min(lo - pad, 0.0) if lo >= 0 else lo - pad, hi + pad)
    ax.set_title(title, fontsize=9); ax.set_xlabel("step"); ax.legend(fontsize=7)


def _plot_cubic(series, vs, suptitle):
    """§10 cubic-approximation panel (4 plots). `vs`='ntk' → compare predictions to the NTK σ₁ and show
    ‖ΔQ‖% as plot 4; `vs`='hess' → compare to the loss-Hessian λmax and show ‖ΔJ‖% as plot 4. Plots 1-3 are
    Eq-47 (single), Eq-51 (multi), and Eq-51 without the η² term (cubic, minus the explicit η² interaction)."""
    actkey = "cActN" if vs == "ntk" else "cActH"
    driftkey = "cdQ" if vs == "ntk" else "cdJ"
    act = series.get(actkey)
    if act is None or not any(y == y for y in act):
        return None
    t = series["t"]
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.3), squeeze=False)
    a = axs[0]
    _cubic_pred_ax(a[0], t, series, "c47", "c47p", act, "Eq-47 (single-sample)", "cubic")
    _cubic_pred_ax(a[1], t, series, "c51", "c51p", act, "Eq-51 (multi-sample)", "cubic")
    _cubic_pred_ax(a[2], t, series, "c51n", "c51np", act, "Eq-51 without the η² term", "cubic (no η²)")
    drift = series.get(driftkey)
    if drift is not None and any(y == y for y in drift):
        a[3].plot(*_finite(t, drift), color="tab:green")
        a[3].set_title("‖ΔQ₍t₊ₛ₎−Q_t‖/‖Q_t‖ × 100" if vs == "ntk" else "‖ΔJ₍t₊ₛ₎−J_t‖/‖J_t‖ × 100",
                       fontsize=9)
        a[3].set_xlabel("step"); a[3].set_ylabel("% over the cubic window")
    else:
        a[3].axis("off")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_section10_ntk(series, meta):
    """§10 panel 1 — cubic-approximation σ₁ predictions (Eq-47/51 ±PSD, Eq-51 w/o η²) vs the empirical
    NTK σ₁, plus the window drift of Q (‖ΔQ‖/‖Q₀‖ %)."""
    return _plot_cubic(series, "ntk",
                       "§10 — cubic-approximation σ₁ predictions vs the empirical NTK σ₁  (plot 4: ‖ΔQ‖ %)")


def plot_section10_hess(series, meta):
    """§10 panel 2 — the same cubic-approximation σ₁ predictions vs the full loss-Hessian sharpness
    λmax(∇²L), plus the window drift of J (‖ΔJ‖/‖J₀‖ %)."""
    return _plot_cubic(series, "hess",
                       "§10 — cubic-approximation σ₁ predictions vs the full-Hessian λmax  (plot 4: ‖ΔJ‖ %)")


_G3D_IDXKEY = {"t1": "i1", "t2": "i2", "t3": "i3"}
_G3D_DKEY = {"t1": "d1", "t2": "d2", "t3": "d3"}     # i=j=k diagonal values, for the highlighted diagonal markers


def _g3d_scatter(fig, pos, g, key, M, title):
    """One 3D scatter of an M×M×M grid: x=i, y=j, z=k (0-based, so the cube starts at the origin and the
    highlighted i=j=k diagonal runs corner-to-corner), colour = signed value.

    The tensors are heavy-tailed (~90% near-zero, both signs), so a linear colour scale washes everything to
    white — we use a SYMLOG norm (linear floor at the 25th percentile of |value|) so the bulk spreads across the
    diverging map while the colorbar still reads true values. Marker size grows with magnitude to declutter the
    cube. `g` is the snapshot dict; when g['sparse'] only the top-|value| points are present (with flat indices)."""
    import numpy as np
    from matplotlib.colors import SymLogNorm
    v = np.asarray(g[key], dtype=float)
    idx = (np.asarray(g[_G3D_IDXKEY[key]], dtype=np.int64) if g.get("sparse")
           else np.arange(M * M * M))                         # flat C-order: idx = i·M² + j·M + k
    i = idx // (M * M); j = (idx // M) % M; k = idx % M
    a = np.abs(v); vmax = max(float(a.max()), 1e-30)
    lin = max(float(np.percentile(a, 25)), vmax * 1e-3, 1e-30)
    c = np.sign(v) * np.log1p(a / lin); cm = max(float(np.abs(c).max()), 1e-30)
    s = 26.0 * (0.35 + 0.65 * np.abs(c) / cm)
    norm = SymLogNorm(linthresh=lin, vmin=-vmax, vmax=vmax)
    ax = fig.add_subplot(*pos, projection="3d")
    sc = ax.scatter(i, j, k, c=v, cmap="RdBu_r", norm=norm,                 # 0-based → cube starts at (0,0,0)
                    s=s, alpha=0.85, depthshade=True, linewidths=0)
    # highlight i=j=k subtly: a faint guide line + lightly-ringed markers that keep their true colour
    d = np.arange(M)
    ax.plot(d, d, d, color="#64748b", lw=1.0, alpha=0.35, zorder=4)
    dv = np.asarray(g.get(_G3D_DKEY[key], []), dtype=float)
    if dv.size == M:
        ax.scatter(d, d, d, c=dv, cmap="RdBu_r", norm=norm, s=30, edgecolors="#475569",
                   linewidths=0.6, depthshade=False, zorder=6)
    ax.set_xlim(M - 1, 0)                                                   # reverse the i-axis (0 at the origin corner, with j,k)
    ax.set_ylim(0, M - 1); ax.set_zlim(0, M - 1)
    ax.set_xlabel("i"); ax.set_ylabel("j"); ax.set_zlabel("k"); ax.set_title(title, fontsize=9)
    return ax, sc


_G3DSQ_LABELS = ["i=j", "i≠j"]
_G3DSQ_COLORS = ["#d62728", "#1f77b4"]
_G3DSQ_TITLE = r"$S=J_i^\top Q_i J_j\,r_j$"


def _g3d_class_counts(M):
    """# of (i,j,k) triples in the 5 classes: i=j=k, i=j≠k, i≠j=k, i=k≠j, i≠j≠k (turns class MEAN→SUM)."""
    import numpy as np
    return np.array([M, M * (M - 1), M * (M - 1), M * (M - 1), M * (M - 1) * (M - 2)], dtype=float)


def _g3dsq_class_counts(M):
    """# of (i,j) pairs in the 2 square classes: i=j, i≠j."""
    import numpy as np
    return np.array([M, M * (M - 1)], dtype=float)


def _g3d_square(fig, pos, g, M, title):
    """2D square S[i,j]=JᵢᵀQᵢJⱼrⱼ — symlog-coloured square-marker scatter, i=j diagonal highlighted."""
    import numpy as np
    from matplotlib.colors import SymLogNorm
    v = np.asarray(g["sq"], dtype=float)
    idx = (np.asarray(g["sqi"], dtype=np.int64) if g.get("sqsparse") else np.arange(M * M))   # idx = i·M + j
    ii = idx // M; jj = idx % M
    a = np.abs(v); vmax = max(float(a.max()), 1e-30)
    lin = max(float(np.percentile(a, 25)), vmax * 1e-3, 1e-30)
    norm = SymLogNorm(linthresh=lin, vmin=-vmax, vmax=vmax)
    c = np.sign(v) * np.log1p(a / lin); cm = max(float(np.abs(c).max()), 1e-30)
    base = max(6.0, min(60.0, 1400.0 / M))
    ax = fig.add_subplot(*pos)
    sc = ax.scatter(ii, jj, c=v, cmap="RdBu_r", norm=norm, s=base * (0.35 + 0.65 * np.abs(c) / cm),
                    marker="s", linewidths=0)
    d = np.arange(M)
    ax.plot(d, d, color="#64748b", lw=1.0, alpha=0.4, zorder=4)          # i=j diagonal guide
    dv = np.asarray(g.get("sqd", []), dtype=float)
    if dv.size == M:
        ax.scatter(d, d, c=dv, cmap="RdBu_r", norm=norm, s=base * 1.2, marker="s",
                   edgecolors="#222", linewidths=0.8, zorder=6)
    ax.set_xlim(-0.5, M - 0.5); ax.set_ylim(-0.5, M - 0.5); ax.set_aspect("equal")
    ax.set_xlabel("i"); ax.set_ylabel("j"); ax.set_title(title, fontsize=9)
    return ax, sc


def _section11_figure(g, M, t):
    """Build the §11 grids figure (3 rotating cubes + the 2D square). Returns (fig, [cube Axes3D])."""
    fig = plt.figure(figsize=(21, 5.2))
    pp = g.get("pp"); cube_axes = []
    for n, (key, lab) in enumerate((("t1", r"$T_1=J_i^\top Q_j J_k$"),
                                    ("t2", r"$T_2=u_j u_k\,J_i^\top Q_j J_k$"),
                                    ("t3", r"$T_3=r_i u_j u_k\,J_i^\top Q_j J_k$"))):
        if pp is not None:
            lab = lab + f"   (+{pp[n]:.1f}%)"                 # positive share 100·Σ(+)/(Σ(+)+|Σ(−)|)
        ax, sc = _g3d_scatter(fig, (1, 4, n + 1), g, key, M, lab); cube_axes.append(ax)
        fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08)
    if "sq" in g:                                            # 4th column: the 2D square (static)
        slab = _G3DSQ_TITLE + (f"   (+{g['sqpp']:.1f}%)" if g.get("sqpp") is not None else "")
        _, sc = _g3d_square(fig, (1, 4, 4), g, M, slab)
        fig.colorbar(sc, ax=fig.axes[-1], shrink=0.55, pad=0.08)
    fig.suptitle(f"§11 — Hessian–NTK grids + square over sample indices (step {t}, N={M})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, cube_axes


def plot_section11(hist):
    """§11 — the three N×N×N Hessian–NTK grids + the 2D square S=JᵢᵀQᵢJⱼrⱼ, at the LAST snapshot (static)."""
    snaps = [r for r in hist if "g3d" in r and "t1" in r["g3d"]]   # grid stored every diagnostic tick
    if not snaps:
        return None
    g = snaps[-1]["g3d"]
    fig, _ = _section11_figure(g, g["M"], snaps[-1]["t"])
    return fig


def save_section11_grids_gif(hist, path, frames=30, fps=12, dpi=62):
    """§11 grids as a rotating GIF (the 3 cubes spin 360° so the 3D structure is readable; square static)."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    snaps = [r for r in hist if "g3d" in r and "t1" in r["g3d"]]
    if not snaps:
        return None
    g = snaps[-1]["g3d"]
    fig, cube_axes = _section11_figure(g, g["M"], snaps[-1]["t"])

    def update(frame):
        az = frame * (360.0 / frames)
        for ax in cube_axes:
            ax.view_init(elev=22, azim=az)
        return []
    FuncAnimation(fig, update, frames=frames, interval=1000 // fps, blit=False).save(
        path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path


def save_section11_evolution_gif(hist, path, key="t3", maxframes=300, fps=10, dpi=70):
    """§11 evolution as a GIF: the chosen grid (default T3) animated across training (with a slow spin).
    The grid is now stored every diagnostic tick, so this shows ONE FRAME PER ITERATION (essential to read
    the Edge-of-Stability oscillation) — only subsampled if the run is longer than maxframes."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    snaps = [r for r in hist if "g3d" in r and "t1" in r["g3d"]]
    if len(snaps) < 2:
        return None
    if len(snaps) > maxframes:
        snaps = [snaps[round(x * (len(snaps) - 1) / (maxframes - 1))] for x in range(maxframes)]
    M = snaps[0]["g3d"]["M"]; lab = {"t1": "T₁", "t2": "T₂", "t3": "T₃"}[key]
    fig = plt.figure(figsize=(7.5, 6.8))

    def update(i):
        fig.clear()
        s = snaps[i]
        ax, _ = _g3d_scatter(fig, (1, 1, 1), s["g3d"], key, M, f"step {s['t']}")
        ax.view_init(elev=22, azim=20 + i * 1.5)
        fig.suptitle(f"§11 — {lab} grid over training (N={M})", fontsize=12)
        return []
    FuncAnimation(fig, update, frames=len(snaps), interval=1000 // fps).save(
        path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path


def plot_section11_evolution(hist, key="t3"):
    """§11 evolution — one grid (default T3) at up to 6 snapshots across training, to show how it develops."""
    snaps = [r for r in hist if "g3d" in r and "t1" in r["g3d"]]   # grid stored every diagnostic tick
    if len(snaps) < 2:
        return None
    pick = [snaps[round(x * (len(snaps) - 1) / 5)] for x in range(6)]
    seen, sel = set(), []
    for s in pick:                                            # de-dup while preserving order (short runs)
        if s["t"] not in seen:
            seen.add(s["t"]); sel.append(s)
    M = sel[0]["g3d"]["M"]
    lab = {"t1": "T₁", "t2": "T₂", "t3": "T₃"}[key]
    fig = plt.figure(figsize=(16.5, 5.4 * ((len(sel) + 2) // 3)))
    for n, s in enumerate(sel):
        _, sc = _g3d_scatter(fig, ((len(sel) + 2) // 3, 3, n + 1), s["g3d"], key, M, f"step {s['t']}")
    fig.suptitle(f"§11 — {lab} grid over the course of training (N={M})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


_G3D_CLASS_LABELS = ["i=j=k", "i=j≠k", "i≠j=k", "i=k≠j", "i≠j≠k"]
_G3D_CLASS_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]


def plot_section11_classes(hist):
    """§11 — evolution of each grid's mean ± std (shaded cloud) over training, for the 4 diagonal classes of
    (i,j,k): i=j=k, i=j≠k, i≠j=k, i≠j≠k. Three panels: T1=JᵢᵀQⱼJₖ, T2=uⱼuₖ·T1, T3=rᵢuⱼuₖ·T1."""
    import numpy as np
    snaps = [r for r in hist if "g3d" in r and "ev" in r["g3d"]]
    if len(snaps) < 2:
        return None
    steps = np.array([r["t"] for r in snaps], dtype=float)
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    titles = [r"$T_1=J_i^\top Q_j J_k$", r"$T_2=u_j u_k\,J_i^\top Q_j J_k$",
              r"$T_3=r_i u_j u_k\,J_i^\top Q_j J_k$"]
    for ti, key in enumerate(("t1", "t2", "t3")):
        ax = axes[ti]
        ev = np.array([r["g3d"]["ev"][key] for r in snaps])   # (T, 5, 3): [...,0]=mean [...,1]=std
        for c in range(5):
            m, s = ev[:, c, 0], ev[:, c, 1]
            ax.fill_between(steps, m - s, m + s, color=_G3D_CLASS_COLORS[c], alpha=0.18, linewidth=0)
            ax.plot(steps, m, color=_G3D_CLASS_COLORS[c], lw=1.4, label=_G3D_CLASS_LABELS[c])
        ax.set_title(titles[ti], fontsize=10); ax.set_xlabel("step"); ax.axhline(0, color="#888", lw=0.6)
        if ti == 0:
            ax.legend(fontsize=8, loc="best", title="class of (i,j,k)", title_fontsize=8)
    ax = axes[3]                                              # 4th column: the square (2 classes)
    if all("sqev" in r["g3d"] for r in snaps):
        sev = np.array([r["g3d"]["sqev"] for r in snaps])     # (T, 2, 3)
        for c in range(2):
            m, s = sev[:, c, 0], sev[:, c, 1]
            ax.fill_between(steps, m - s, m + s, color=_G3DSQ_COLORS[c], alpha=0.18, linewidth=0)
            ax.plot(steps, m, color=_G3DSQ_COLORS[c], lw=1.4, label=_G3DSQ_LABELS[c])
        ax.legend(fontsize=8, loc="best", title="class of (i,j)", title_fontsize=8)
    ax.set_title(_G3DSQ_TITLE, fontsize=10); ax.set_xlabel("step"); ax.axhline(0, color="#888", lw=0.6)
    fig.suptitle("§11 — per-class mean ± std of the Hessian–NTK grids + square over training", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_section11_sumevolution(hist):
    """§11 — evolution of each class's SUM over training (no std): s_c = mean_c·count_c. 4 columns (cubes + square)."""
    import numpy as np
    snaps = [r for r in hist if "g3d" in r and "ev" in r["g3d"]]
    if len(snaps) < 2:
        return None
    steps = np.array([r["t"] for r in snaps], dtype=float)
    M = snaps[0]["g3d"]["M"]; ct5 = _g3d_class_counts(M); ct2 = _g3dsq_class_counts(M)
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    titles = [r"$T_1=J_i^\top Q_j J_k$", r"$T_2=u_j u_k\,J_i^\top Q_j J_k$",
              r"$T_3=r_i u_j u_k\,J_i^\top Q_j J_k$"]
    for ti, key in enumerate(("t1", "t2", "t3")):
        ax = axes[ti]
        s = np.array([[r["g3d"]["ev"][key][c][0] for c in range(5)] for r in snaps]) * ct5
        for c in range(5):
            ax.plot(steps, s[:, c], color=_G3D_CLASS_COLORS[c], lw=1.4, label=_G3D_CLASS_LABELS[c])
        ax.set_title(titles[ti], fontsize=10); ax.set_xlabel("step"); ax.set_ylabel("sum"); ax.axhline(0, color="#888", lw=0.6)
        if ti == 0:
            ax.legend(fontsize=8, loc="best", title="class of (i,j,k)", title_fontsize=8)
    ax = axes[3]                                              # 4th column: the square (2 classes)
    if all("sqev" in r["g3d"] for r in snaps):
        s = np.array([[r["g3d"]["sqev"][c][0] for c in range(2)] for r in snaps]) * ct2
        for c in range(2):
            ax.plot(steps, s[:, c], color=_G3DSQ_COLORS[c], lw=1.4, label=_G3DSQ_LABELS[c])
        ax.legend(fontsize=8, loc="best", title="class of (i,j)", title_fontsize=8)
    ax.set_title(_G3DSQ_TITLE, fontsize=10); ax.set_xlabel("step"); ax.set_ylabel("sum"); ax.axhline(0, color="#888", lw=0.6)
    fig.suptitle("§11 — per-class SUM of the Hessian–NTK grids + square over training", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def _plot_section11_share(hist, weighted, suptitle):
    """Shared body for the two norm-share panels. weighted=False → class MEAN (100·m_c²/Σm²);
    weighted=True → class SUM (100·s_c²/Σs², s_c=mean_c·count_c). 4th column = the square (2 classes)."""
    import numpy as np
    snaps = [r for r in hist if "g3d" in r and "ev" in r["g3d"]]
    if len(snaps) < 2:
        return None
    steps = np.array([r["t"] for r in snaps], dtype=float)
    M = snaps[0]["g3d"]["M"]
    ct5 = _g3d_class_counts(M) if weighted else np.ones(5)
    ct2 = _g3dsq_class_counts(M) if weighted else np.ones(2)
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    titles = [r"$T_1=J_i^\top Q_j J_k$", r"$T_2=u_j u_k\,J_i^\top Q_j J_k$",
              r"$T_3=r_i u_j u_k\,J_i^\top Q_j J_k$"]
    for ti, key in enumerate(("t1", "t2", "t3")):
        ax = axes[ti]
        x = np.array([[r["g3d"]["ev"][key][c][0] for c in range(5)] for r in snaps]) * ct5
        sq = x ** 2; den = sq.sum(axis=1, keepdims=True); den[den == 0] = 1.0
        pct = 100.0 * sq / den
        for c in range(5):
            ax.plot(steps, pct[:, c], color=_G3D_CLASS_COLORS[c], lw=1.4, label=_G3D_CLASS_LABELS[c])
        ax.set_title(titles[ti], fontsize=10); ax.set_xlabel("step"); ax.set_ylabel("% of norm"); ax.set_ylim(0, 100)
        if ti == 0:
            ax.legend(fontsize=8, loc="best", title="class of (i,j,k)", title_fontsize=8)
    ax = axes[3]                                              # 4th column: the square (2 classes)
    if all("sqev" in r["g3d"] for r in snaps):
        x = np.array([[r["g3d"]["sqev"][c][0] for c in range(2)] for r in snaps]) * ct2
        sq = x ** 2; den = sq.sum(axis=1, keepdims=True); den[den == 0] = 1.0
        pct = 100.0 * sq / den
        for c in range(2):
            ax.plot(steps, pct[:, c], color=_G3DSQ_COLORS[c], lw=1.4, label=_G3DSQ_LABELS[c])
        ax.legend(fontsize=8, loc="best", title="class of (i,j)", title_fontsize=8)
    ax.set_title(_G3DSQ_TITLE, fontsize=10); ax.set_xlabel("step"); ax.set_ylabel("% of norm"); ax.set_ylim(0, 100)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_section11_normshare(hist):
    """§11 — % of norm contributed by each class MEAN: 100·m_c²/Σ m_c'² (curves sum to 100% per panel)."""
    return _plot_section11_share(hist, False, "§11 — % of norm contributed by each class MEAN  (100·mₖ²/Σmₖ²)")


def plot_section11_sumnormshare(hist):
    """§11 — % of norm contributed by each class SUM: 100·s_c²/Σ s_c'², s_c=mean_c·count_c (size-weighted)."""
    return _plot_section11_share(hist, True, "§11 — % of norm contributed by each class SUM  (100·sₖ²/Σsₖ²)")


# ─────────────────────────────── §12 per-sample Hessian eigenvector cross-similarity ───────────────────────


def _s12_heat(ax, vals, N, title):
    """2D (i,j) grid heatmap; diverging RdBu_r for signed values, sequential viridis for non-negative."""
    import numpy as np
    A = np.asarray(vals, dtype=float).reshape(N, N)
    if A.min() < -1e-9:
        m = max(float(np.abs(A).max()), 1e-30)
        im = ax.imshow(A, cmap="RdBu_r", vmin=-m, vmax=m, origin="upper")
    else:
        im = ax.imshow(A, cmap="viridis", vmin=0.0, vmax=max(float(A.max()), 1e-30), origin="upper")
    ax.set_xlabel("j"); ax.set_ylabel("i"); ax.set_title(title, fontsize=9)
    return im


# §12 5-grid panels: A=m·m (non-neg); B,C signed [-1,1]; D,E signed (σ_j-scaled). Each is N×N×N over (i,j,k).
_S12_GRIDS = [("A", "A · m·m", False), ("B", "B · signed σ,J", True),
              ("C", "C · signed σ,r,J", True), ("D", "D · cc·σ_j", True), ("E", "E · cc·r·σ_j", True)]


def _s12_grid3d(fig, pos, pack, N, title, signed):
    """3D scatter of an N×N×K grid over (i,j,k) from a packed grid {v,idx,d}; i=j=k diagonal ringed."""
    import numpy as np
    from matplotlib.colors import Normalize
    v = np.asarray(pack["v"], dtype=float)
    q = np.asarray(pack["idx"]) if pack.get("idx") is not None else np.arange(v.size)
    i = q // (N * N); j = (q // N) % N; k = q % N
    cmap = "RdBu_r" if signed else "viridis"
    m = max(float(np.abs(v).max()), 1e-30)
    norm = Normalize(-m, m) if signed else Normalize(0.0, m)
    ax = fig.add_subplot(*pos, projection="3d")
    sc = ax.scatter(i, j, k, c=v, cmap=cmap, norm=norm, s=10, alpha=0.7, depthshade=True, linewidths=0)
    dg = np.arange(N); d = np.asarray(pack["d"], dtype=float)
    ax.scatter(dg, dg, dg, c=d, cmap=cmap, norm=norm, s=55, edgecolors="k", linewidths=0.6)   # i=j=k highlight
    ax.set_xlim(N - 1, 0); ax.set_ylim(0, N - 1); ax.set_zlim(0, N - 1)
    ax.set_xlabel("i"); ax.set_ylabel("j"); ax.set_zlabel("k"); ax.set_title(title, fontsize=8)
    return ax, sc


def _runmean(snaps, fn):
    """time-running mean of fn(g4d) over snapshots (skips non-finite)."""
    import math
    vs = [fn(r["g4d"]) for r in snaps]
    vs = [v for v in vs if v is not None and math.isfinite(v)]
    return sum(vs) / len(vs) if vs else float("nan")


def _s12_grids_fig(snaps, k0):
    """One §12 panel (k0=1/2/5): 5 3D grids A,B,C,D,E over (i,j,k). Title = current μ + running ⟨μ⟩."""
    g = snaps[-1]["g4d"]; N = g["M"]; S = g["s12"][str(k0)]; key = str(k0)
    fig = plt.figure(figsize=(22, 4.2))
    for n, (nm, lab, signed) in enumerate(_S12_GRIDS):
        pk = S[nm]; run = _runmean(snaps, lambda gg: gg["s12"][key][nm]["mn"])
        ax, sc = _s12_grid3d(fig, (1, 5, n + 1), pk, N, f"{lab}  μ={pk['mn']:.2f}  ⟨μ⟩={run:.2f}", signed)
        fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.06)
    fig.suptitle(f"§12 — per-sample Hessian triple-(i,j,k) grids (k0={k0})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def _s12_3d_snaps(hist):
    return [r for r in hist if "g4d" in r and r["g4d"].get("do3d")]


def plot_section12_panel1(hist):
    s = _s12_3d_snaps(hist); return _s12_grids_fig(s, 1) if s else None


def plot_section12_panel2(hist):
    s = _s12_3d_snaps(hist); return _s12_grids_fig(s, 2) if s else None


def plot_section12_panel3(hist):
    s = _s12_3d_snaps(hist); return _s12_grids_fig(s, 5) if s else None


def plot_section12_angles(hist):
    """§12 panel 1 — four (i,j) grids of the MEAN principal angle between Qᵢ,Qⱼ subspaces (k=1,5,10,kfull).
    ang.g now has 5 grids (k=1,2,5,10,kfull); panel 1 shows k=1,5,10,kfull → grid indices [0,2,3,4] (skip k=2)."""
    import numpy as np
    snaps = [r for r in hist if "g4d" in r]
    if not snaps:
        return None
    g = snaps[-1]["g4d"]; N = g["M"]; P = g["ang"]
    idxs = [0, 2, 3, 4]; labs = ["k=1", "k=5", "k=10", f"k={P.get('kfull', 15)}"]   # skip index 1 (k=2)
    fig = plt.figure(figsize=(16, 3.7))
    for n, idx in enumerate(idxs):
        ax = fig.add_subplot(1, 4, n + 1)
        A = np.asarray(P["g"][idx], dtype=float).reshape(N, N)
        im = ax.imshow(A, cmap="magma", vmin=0.0, vmax=90.0, origin="upper")
        run = _runmean(snaps, lambda gg, i=idx: gg["ang"]["gm"][i])
        ax.set_xlabel("j"); ax.set_ylabel("i"); ax.set_title(f"angle {labs[n]}   μ={P['gm'][idx]:.1f}°  ⟨μ⟩={run:.1f}°", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("§12 panel 1 — mean principal angle between Qᵢ,Qⱼ subspaces (deg)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# §12 panels 2/3 — principal-angle evolution over OFF-DIAGONAL pairs (i≠j). ang.{mn,mx,me} per grid index.
_S12_KIDX = {1: 0, 2: 1, 5: 2, 10: 3, 15: 4}            # k → ang grid index (15 = kfull slot, index 4)
_S12_EVO_KS = [5, 15, 1, 2]                              # row order
_S12_EVO_STATS = [("mn", "min", "#dc2626"), ("mx", "max", "#2563eb"), ("me", "mean", "#16a34a")]


def plot_section12_evolution(hist):
    """§12 panels 2/3 — min/max/mean principal angle over OFF-DIAGONAL pairs (i≠j) vs step. 4×3 grid:
    rows = k∈[5,15,1,2], cols = [min,max,mean]. Reads ang.{mn,mx,me}[idx] across §12 records."""
    import numpy as np
    snaps = [r for r in hist if "g4d" in r]   # angles exist for every snapshot (not gated by the 3D-grid cap)
    if len(snaps) < 2:
        return None
    t = [r["t"] for r in snaps]
    fig, axs = plt.subplots(4, 3, figsize=(14, 12))
    for ri, k in enumerate(_S12_EVO_KS):
        idx = _S12_KIDX[k]
        for ci, (sk, slab, col) in enumerate(_S12_EVO_STATS):
            ax = axs[ri, ci]
            a = np.array([r["g4d"]["ang"][sk][idx] for r in snaps], dtype=float)
            ax.plot(t, a, color=col)
            ax.set_xlabel("step"); ax.set_ylabel("angle (deg)")
            ax.set_title(f"{slab} principal angle  k={k}", fontsize=10)
    fig.suptitle("§12 panels 2/3 — principal-angle evolution over off-diagonal pairs (i≠j)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


_S12_DAL_KS = [1, 2, 5, 15]


def plot_section12_diagalign(hist):
    """§12a panel 4 — diagonal (same-rank) eigenvector alignment mean_{i≠j}(1/2k)Σ_a|⟨e_{i,a},e_{j,a}⟩| ∈ [0,1] vs
    step, one line per k∈{1,2,5,15}. Reads ang.dal. None if <2 records."""
    import numpy as np
    snaps = [r for r in hist if "g4d" in r and "dal" in r["g4d"].get("ang", {})]
    if len(snaps) < 2:
        return None
    t = [r["t"] for r in snaps]
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.6))
    for i, k in enumerate(_S12_DAL_KS):
        ax = axs[i]
        y = np.array([r["g4d"]["ang"]["dal"][i] for r in snaps], dtype=float)
        ax.plot(t, y, color="#0891b2", lw=1.6)
        ax.set_ylim(-0.02, 1.02); ax.set_xlabel("step"); ax.set_ylabel("alignment ∈ [0,1]")
        ax.set_title(f"mean |diag| alignment  k={k}", fontsize=10)
    fig.suptitle("§12a panel 4 — diagonal (same-rank) eigenvector alignment over off-diagonal pairs", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


_S12_XP_TITLES = ["mean_i Σ_j |<J_i,u_j>|·σ_i·r_i", "mean_i Σ_j |<J_i,u_j>||<J_j,u_i>|·σ_iσ_j·r_i r_j",
                  "mean_{i≠j} |<J_i,u_j>|/||J_i||", "mean_{i≠j} |<J_i,u_j>||<J_j,u_i>|/(||J_i|| ||J_j||)"]


def plot_section12_xproj(hist):
    """§12b panels 8/9 — SVD-principal-direction → nearest-eigenvector cross projections, MIN (row 0) / MAX (row 1)
    principal-angle direction; 4 means over ALL (i,j) vs step. Reads xproj.{min,max}. None if <2 records."""
    import numpy as np
    snaps = [r for r in hist if "g4d" in r and "xproj" in r["g4d"]]
    if len(snaps) < 2:
        return None
    t = [r["t"] for r in snaps]
    rows = [("min", "#2563eb"), ("max", "#dc2626")]
    fig, axs = plt.subplots(2, 4, figsize=(20, 7))
    for ri, (key, col) in enumerate(rows):
        for ci in range(4):
            ax = axs[ri][ci]
            y = np.array([r["g4d"]["xproj"][key][ci] for r in snaps], dtype=float)
            ax.axhline(0, c="#d1d5db", lw=0.8)
            ax.plot(t, y, color=col, marker="o", ms=3, lw=1.4)
            ax.set_xlabel("step"); ax.set_title(f"{key} · mean {_S12_XP_TITLES[ci]}", fontsize=8)
    fig.suptitle("§12b panels 8/9 — cross projections at the min/max principal-angle direction", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


S12_HISTBINS = 40   # §12 panel-4/5 alignment-histogram bar count (shared bins across the two ranks) — MIRRORS index.html.


def _s12_proj_fig(hist, dkey, row_titles, suptitle, ulab="u"):
    """Shared body for the §12b projection time-series panels. dkey='proj' (per-sample Q_i, eigvec label u) or
    'gproj' (AVERAGED (1/N)Σ_i Q_i, eigvec label w). PRODUCT |⟨J,·⟩|·σ·r (cols 1-2) and ALIGNMENT |⟨J,·⟩|/‖J‖ ∈ [0,1]
    (cols 3-4); TOP-2 (row 0) and BOTTOM-2 (row 1); rank-1 solid, rank-2 dashed. None if <2 records."""
    import numpy as np
    recs = [r for r in hist if "g4d" in r and r["g4d"].get(dkey) is not None
            and r["g4d"][dkey].get("top")]
    if len(recs) < 2:
        return None
    t = list(range(len(recs)))
    P = [r["g4d"][dkey] for r in recs]
    cols = [("prod", 0, "mean of proj·r", False), ("prod", 1, "std of proj·r", False),
            ("cos", 0, f"mean of |⟨J,{ulab}⟩|/‖J‖", True), ("cos", 1, f"std of |⟨J,{ulab}⟩|/‖J‖", False)]
    rows = [("top", row_titles[0], ["#2563eb", "#93c5fd"]),
            ("bot", row_titles[1], ["#dc2626", "#fca5a5"])]
    nrank = max((len(p.get("top", [])) for p in P), default=0)
    rlab = ["rank-1", "rank-2"]; dash = ["-", "--"]
    fig, axs = plt.subplots(2, 4, figsize=(20, 8))
    for ri, (grp, rlabel, palette) in enumerate(rows):
        for ci, (key, si, yl, pin) in enumerate(cols):
            ax = axs[ri][ci]
            ax.axhline(0, c="#d1d5db", lw=0.8)
            for rk in range(nrank):
                y = np.array([(p[grp][rk][key][si] if rk < len(p.get(grp, [])) else np.nan) for p in P], dtype=float)
                ax.plot(t, y, color=palette[rk % len(palette)], marker="o", ms=3, lw=1.4,
                        ls=dash[rk % len(dash)], label=(rlab[rk] if rk < len(rlab) else f"rank-{rk + 1}"))
            ax.set_xlabel("step"); ax.set_ylabel(yl)
            if pin:                                            # mean alignment ∈ [0,1]; pin the y-axis so the scale is honest
                ax.set_ylim(-0.02, 1.02)
            ax.set_title((rlabel + " · " if ci == 0 else "") + yl, fontsize=9)
            if ci in (0, 2) and nrank > 1:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_section12_proj(hist):
    """§12b panels 1/2 — per-sample PRODUCT |⟨J_i,u⟩|·σ·r_i & ALIGNMENT |⟨J_i,u⟩|/‖J_i‖ for the TOP-2 (row 0)
    and BOTTOM-2 (row 1) eigenvectors u of EACH per-sample Q_i. Reads proj.{top,bot}."""
    return _s12_proj_fig(hist, "proj",
        ["Panel 1 — per-sample top-2 u_{i,1},u_{i,2}", "Panel 2 — per-sample bottom-2 u_{i,-1},u_{i,-2}"],
        "§12b panels 1/2 — per-sample product |⟨J,u⟩|·σ·r (cols 1-2) & alignment |⟨J,u⟩|/‖J‖ ∈ [0,1] (cols 3-4)", "u")


def plot_section12_gproj(hist):
    """§12b panels 3/4 — AVERAGED view: each ∇f_i projected onto the TOP-2 (row 0) / BOTTOM-2 (row 1) eigenvectors
    w of the AVERAGED Hessian (1/N)Σ_i Q_i (one shared basis); product uses its eigenvalue σ. Reads gproj.{top,bot}.
    Empty unless the averaged-view eigendecomposition ran (s22). At N=1 it coincides with panels 1/2."""
    return _s12_proj_fig(hist, "gproj",
        ["Panel 3 — AVERAGED top-2 w_1,w_2 of (1/N)Σ_i Q_i", "Panel 4 — AVERAGED bottom-2 w_{-1},w_{-2} of (1/N)Σ_i Q_i"],
        "§12b panels 3/4 — AVERAGED view onto (1/N)Σ_i Q_i eigvecs: product |⟨J,w⟩|·σ·r (cols 1-2) & alignment |⟨J,w⟩|/‖J‖ (cols 3-4)", "w")


def _s12_wproj_fig(hist, pi, title):
    """One §12b WEIGHTED-AVERAGE panel (5/6/7, pi∈{0,1,2}) — 2 rows: TOP-2 (row 0) & BOTTOM-2 (row 1) eigenvectors of
    the weighted-average Hessian (Σ_i w_i Q_i)/(Σ_i w_i). Reads wproj[pi].{top,bot}. None if <2 records / panel absent."""
    import numpy as np
    recs = [r for r in hist if "g4d" in r and r["g4d"].get("wproj") and len(r["g4d"]["wproj"]) > pi
            and r["g4d"]["wproj"][pi].get("top")]
    if len(recs) < 2:
        return None
    t = list(range(len(recs)))
    P = [r["g4d"]["wproj"][pi] for r in recs]
    cols = [("prod", 0, "mean of proj·r", False), ("prod", 1, "std of proj·r", False),
            ("cos", 0, "mean of |⟨J,e⟩|/‖J‖", True), ("cos", 1, "std of |⟨J,e⟩|/‖J‖", False)]
    rows = [("top", "Row 1 — TOP-2 eigvecs of ⟨Q⟩_w", ["#2563eb", "#93c5fd"]),
            ("bot", "Row 2 — BOTTOM-2 eigvecs of ⟨Q⟩_w", ["#dc2626", "#fca5a5"])]
    nrank = max((len(p.get("top", [])) for p in P), default=0)
    rlab = ["rank-1", "rank-2"]; dash = ["-", "--"]
    fig, axs = plt.subplots(2, 4, figsize=(20, 8))
    for ri, (grp, rlabel, palette) in enumerate(rows):
        for ci, (key, si, yl, pin) in enumerate(cols):
            ax = axs[ri][ci]
            ax.axhline(0, c="#d1d5db", lw=0.8)
            for rk in range(nrank):
                y = np.array([(p[grp][rk][key][si] if rk < len(p.get(grp, [])) else np.nan) for p in P], dtype=float)
                ax.plot(t, y, color=palette[rk % len(palette)], marker="o", ms=3, lw=1.4,
                        ls=dash[rk % len(dash)], label=(rlab[rk] if rk < len(rlab) else f"rank-{rk + 1}"))
            ax.set_xlabel("step"); ax.set_ylabel(yl)
            if pin:
                ax.set_ylim(-0.02, 1.02)
            ax.set_title((rlabel + " · " if ci == 0 else "") + yl, fontsize=9)
            if ci in (0, 2) and nrank > 1:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_section12_wproj5(hist):
    """§12b Panel 5 — weighted-average Hessian, weights = |top NTK eigenvector|."""
    return _s12_wproj_fig(hist, 0, "§12b Panel 5 — weighted avg ⟨Q⟩_w, weights w_i = |top NTK eigvec component|")


def plot_section12_wproj6(hist):
    """§12b Panel 6 — weighted-average Hessian, weights = |2nd NTK eigenvector|."""
    return _s12_wproj_fig(hist, 1, "§12b Panel 6 — weighted avg ⟨Q⟩_w, weights w_i = |2nd NTK eigvec component|")


def plot_section12_wproj7(hist):
    """§12b Panel 7 — weighted-average Hessian, weights = random U[0,1] (unit-normalized)."""
    return _s12_wproj_fig(hist, 2, "§12b Panel 7 — weighted avg ⟨Q⟩_w, weights w_i = random U[0,1] (unit-norm)")


def _s12_hist_fig(hist, dkey, suptitle, ulab="u"):
    """Shared body for the §12b over-sample histograms at the LAST snapshot. dkey='proj' (per-sample, label u) or
    'gproj' (GLOBAL, label w). PRODUCT proj_i·r_i (col 0) and ALIGNMENT |⟨J_i,·⟩|/‖J_i‖ (col 1) for TOP-2 (row 0)
    and BOTTOM-2 (row 1); rank-1 & rank-2 overlaid (shared bins). Reads {dkey}.{top,bot}=[{pvals,vals},…]."""
    import numpy as np
    def has_vals(r):
        tp = r["g4d"][dkey].get("top") if r.get("g4d", {}).get(dkey) else None
        return bool(tp) and "vals" in tp[0]
    recs = [r for r in hist if "g4d" in r and r["g4d"].get(dkey) is not None and has_vals(r)]
    if not recs:
        return None
    P = recs[-1]["g4d"][dkey]; step = recs[-1].get("t", len(recs) - 1)
    rows = [("top", ["#2563eb", "#93c5fd"]), ("bot", ["#dc2626", "#fca5a5"])]
    hcols = [(0, "pvals", "proj·r", "proj_i·r_i"), (1, "vals", f"|⟨J,{ulab}⟩|/‖J‖", f"|⟨J_i,{ulab}⟩|/‖J_i‖")]
    rlab = ["rank-1", "rank-2"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 7))
    for ri, (grp, palette) in enumerate(rows):
        ranks = P.get(grp, []) or []
        for ci, key, qname, xl in hcols:
            ax = axs[ri][ci]
            allv = np.concatenate([np.asarray(rk.get(key, []), dtype=float) for rk in ranks]) if ranks else np.array([])
            allv = allv[np.isfinite(allv)]
            if allv.size:                                      # shared bins across ranks so the overlay aligns
                lo, hi = float(allv.min()), float(allv.max())
                if hi == lo:
                    lo, hi = lo - 0.5, hi + 0.5
                vbins = np.linspace(lo, hi, S12_HISTBINS + 1)
            else:
                vbins = 10
            for rk, rkd in enumerate(ranks):
                v = np.asarray(rkd.get(key, []), dtype=float)
                lab = rlab[rk] if rk < len(rlab) else f"rank-{rk + 1}"
                ax.hist(v, bins=vbins, color=palette[rk % len(palette)], edgecolor="white", alpha=0.6, label=lab)
            ax.set_title(f"{grp} · {qname} over samples (step {step})", fontsize=10)
            ax.set_xlabel(xl); ax.set_ylabel("count"); ax.legend(fontsize=7)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_section12_hist(hist):
    """§12b panels 1/2 — per-sample product & alignment histograms at the last iteration (per-sample Q_i eigvecs u)."""
    return _s12_hist_fig(hist, "proj",
        "§12b panels 1/2 — per-sample product & alignment histograms at the last iteration", "u")


def plot_section12_ghist(hist):
    """§12b panels 3/4 — AVERAGED-view product & alignment histograms at the last iteration ((1/N)Σ_i Q_i eigvecs w)."""
    return _s12_hist_fig(hist, "gproj",
        "§12b panels 3/4 — AVERAGED-view product & alignment histograms at the last iteration", "w")


# ─────────────────── §13 G1/G2/G3 approximations of the exact reference J_iᵀQ_jJ_k·r_k ───────────────────
# TIME-SERIES of mean±std over classes of triples (i,j,k): the EXACT reference (per-sample HVPs, §11's tool)
# vs each rank-k₀ approximation G1/G2/G3 (built from the per-sample Lanczos top-k₀⊕bottom-k₀ eigenpairs).
_S13_DOMS = [("ndist", "i≠j≠k"), ("ijeq", "i=j≠k"), ("diag", "i=j=k")]   # plotted column order
_S13_KS = [2, 5]                                                          # rows = k₀


def _s13_recs(hist):
    """§13 records (those carrying a 'g13' block) — one per eig-tick after the first."""
    return [r for r in hist if "g13" in r]


def _s13_meanstd(series):
    """split a list of [mean,std] pairs into two numpy arrays (mean, std)."""
    import numpy as np
    arr = np.asarray(series, dtype=float)
    return arr[:, 0], arr[:, 1]


def _plot_section13_panel(hist, n):
    """§13 panel n (n∈{1,2,3}) — Gn vs the exact reference, ONE row of 6 subplots (k₀∈{2,5} × domains).
    Each subplot overlays two mean±std clouds vs step t: the reference (black) and Gn (blue)."""
    import numpy as np
    g13recs = _s13_recs(hist)
    if len(g13recs) < 2:
        return None
    t = [r["t"] for r in g13recs]
    gkey = "g" + str(n)
    cells = [(k0, dom, domlabel) for k0 in _S13_KS for (dom, domlabel) in _S13_DOMS]   # k₀-major: 2×{doms} then 5×{doms}
    fig, axs = plt.subplots(1, 6, figsize=(24, 3.6))
    for ax, (k0, dom, domlabel) in zip(axs, cells):
        rm, rs = _s13_meanstd([r["g13"]["ref"][dom] for r in g13recs])           # reference (same in all 6)
        ax.plot(t, rm, color="#111", lw=1.4, label="ref")
        ax.fill_between(t, rm - rs, rm + rs, color="#999", alpha=0.15)
        gm, gs = _s13_meanstd([r["g13"][gkey][str(k0)][dom] for r in g13recs])   # Gn approximation
        ax.plot(t, gm, color="#2563eb", lw=1.4, label=f"G{n}")
        ax.fill_between(t, gm - gs, gm + gs, color="#2563eb", alpha=0.16)
        ax.set_xlabel("step"); ax.set_title(f"{domlabel}  (k₀={k0})", fontsize=9)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"§13 panel {n} — G{n} vs reference J_iᵀQ_jJ_k·r_k", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_section13_panel1(hist):
    return _plot_section13_panel(hist, 1)


def plot_section13_panel2(hist):
    return _plot_section13_panel(hist, 2)


def plot_section13_panel3(hist):
    return _plot_section13_panel(hist, 3)


def plot_section13_ndiff(hist):
    """§13 panel 4 — residual-reweighting asymmetry ‖A−B‖/‖A‖·100 (%) over training, where
    A=(Σ_i r_{i,t+1}Q_{i,t})(Σ_j J_{j,t}r_{j,t}), B=(Σ_i r_{i,t}Q_{i,t})(Σ_j J_{j,t}r_{j,t+1}). None if no data."""
    recs = [r for r in _s13_recs(hist) if "ndiff" in r["g13"]]
    if len(recs) < 1:
        return None
    t = [r["t"] for r in recs]; y = [r["g13"]["ndiff"] for r in recs]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, y, color="#7c3aed", marker="o", ms=3, lw=1.5)
    ax.axhline(0, c="#d1d5db", lw=0.8)
    ax.set_xlabel("step"); ax.set_ylabel("%")
    ax.set_title("§13 residual-reweighting asymmetry  ‖A−B‖/‖A‖ × 100", fontsize=11)
    fig.tight_layout()
    return fig


# ── §14: per-triplet (i,j,k) decomposition of Tr(ΔNTK). 10 panels of N×N×N cubes (μ / μ⁺ / μ⁻ titles). ──
_S14_AGG = ["max", "min", "Σtop-y", "Σbot-y", "Σall"]
_S14_SUB = ["a", "b·d", "a·b·d", "a·b·d·e", "a·e"]
_S14_RLAB = ["y1·first", "y2·first", "y5·first", "y1·last", "y2·last", "y5·last"]


def _s14_snaps(hist):
    return [r for r in hist if "g14" in r and r["g14"].get("do3d")]


def _s14_fig(packs, labels, N, suptitle):
    ncol = len(packs)
    fig = plt.figure(figsize=(3.5 * ncol, 4.1))
    for n, (pk, lab) in enumerate(zip(packs, labels)):
        if pk is None:
            ax = fig.add_subplot(1, ncol, n + 1, projection="3d"); ax.set_title(lab + "\n(n/a)", fontsize=8); continue
        ax, sc = _s12_grid3d(fig, (1, ncol, n + 1), pk, N,
                             f"{lab}\nμ={pk['mn']:.1e} μ⁺={pk['mp']:.1e} μ⁻={pk['mng']:.1e}", True)
        fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.06)
    fig.suptitle(suptitle, fontsize=11); fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def plot_section14(hist, y, grp):
    """§14 panel for rank-2y and group ∈ {agg, max, min} (5 cubes)."""
    snaps = _s14_snaps(hist)
    if not snaps:
        return None
    g = snaps[-1]["g14"]; N = g["M"]
    P = g["p14"][str(y)][grp]
    labs = _S14_AGG if grp == "agg" else _S14_SUB
    title = {"agg": "sorted-term aggregates", "max": "MAX-element sub-products",
             "min": "MIN-element sub-products"}[grp]
    return _s14_fig(P, labs, N, f"§14  y={y} — {title}")


def plot_section14_ratio(hist):
    """§14 panel 10 — eq-③ multi-step ratio over the residual history (first/last × y, S=y; 6 cubes)."""
    snaps = _s14_snaps(hist)
    if not snaps:
        return None
    g = snaps[-1]["g14"]; N = g["M"]
    return _s14_fig(g["ratio"], _S14_RLAB, N, "§14 panel 10 — eq-③ multi-step ratio (first/last × y, S=y)")


def save_section12_panel_gif(hist, k0, path, frames=30, fps=12, dpi=60):
    """A §12 grid panel as a rotating GIF (the 3D (i,j,k) grids spin so the structure is readable)."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    snaps = _s12_3d_snaps(hist)
    if not snaps:
        return None
    fig = _s12_grids_fig(snaps[-1]["g4d"], k0)
    cube_axes = [ax for ax in fig.axes if hasattr(ax, "view_init")]

    def update(frame):
        az = frame * (360.0 / frames)
        for ax in cube_axes:
            ax.view_init(elev=22, azim=az)
        return []
    FuncAnimation(fig, update, frames=frames, interval=1000 // fps, blit=False).save(
        path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path


def save_panels(results, outdir):
    """Render every supported section to PNGs in `outdir`. Returns the list of files written."""
    os.makedirs(outdir, exist_ok=True)
    series, meta, hist = results["series"], results["meta"], results["history"]
    figs = {"section1_loss_sharpness": plot_section1(series, meta),
            "section2_top_eig": plot_section2(series),
            "section3_bottom_eig": plot_section3(series),
            "section4_J_onto_H": plot_section4(series),
            "section5_slq": plot_slq(hist, meta.get("nsamp", 1)),
            "section5_trace": plot_section5_trace(hist),
            "section6_rotation": plot_section6(series),
            "section7a_ntk_alignment": plot_section7a(series),
            "section7_ntk_fh_svd": plot_section7(series),
            "section8_fh_reshape_svd": plot_section8(series),
            "section4b_Jr_onto_Q": plot_section4b(series),
            "section4c_QJr_onto_GN": plot_section4c(series),
            "section4d_sign_groups": plot_section4d(series),
            "section9_theory_vs_empirical": plot_section9(series, meta, show_psd=False,
                                                          suptitle="§9 — predicted σ₁ vs empirical"),
            "section9b_psd_vs_empirical": plot_section9b(series, meta),
            "section9c_theory_vs_full_hessian": plot_section9c(series, meta),
            "section9d_selfresidual_vs_empirical": plot_section9d(series, meta),
            "section9dc_selfresidual_vs_full_hessian": plot_section9dc(series, meta),
            "section10_cubic_vs_ntk": plot_section10_ntk(series, meta),
            "section10_cubic_vs_full_hessian": plot_section10_hess(series, meta),
            "section11_classes": plot_section11_classes(hist),
            "section11_sumevolution": plot_section11_sumevolution(hist),
            "section11_normshare": plot_section11_normshare(hist),
            "section11_sumnormshare": plot_section11_sumnormshare(hist),
            "section12_panel1_angles": plot_section12_angles(hist),
            "section12_panel2_evolution": plot_section12_evolution(hist),
            "section12a_panel4_diagalign": plot_section12_diagalign(hist),
            "section12b_panel89_xproj": plot_section12_xproj(hist),
            "section12b_panel12_proj": plot_section12_proj(hist),
            "section12b_panel12_hist": plot_section12_hist(hist),
            "section12b_panel34_gproj": plot_section12_gproj(hist),
            "section12b_panel34_ghist": plot_section12_ghist(hist),
            "section12b_panel5_wproj_topNTK": plot_section12_wproj5(hist),
            "section12b_panel6_wproj_2ndNTK": plot_section12_wproj6(hist),
            "section12b_panel7_wproj_random": plot_section12_wproj7(hist),
            "section13_panel1_G1": plot_section13_panel1(hist),
            "section13_panel2_G2": plot_section13_panel2(hist),
            "section13_panel3_G3": plot_section13_panel3(hist),
            "section13_panel4_ndiff": plot_section13_ndiff(hist),
            "section14_y1_agg": plot_section14(hist, 1, "agg"),
            "section14_y1_max": plot_section14(hist, 1, "max"),
            "section14_y1_min": plot_section14(hist, 1, "min"),
            "section14_y2_agg": plot_section14(hist, 2, "agg"),
            "section14_y2_max": plot_section14(hist, 2, "max"),
            "section14_y2_min": plot_section14(hist, 2, "min"),
            "section14_y5_agg": plot_section14(hist, 5, "agg"),
            "section14_y5_max": plot_section14(hist, 5, "max"),
            "section14_y5_min": plot_section14(hist, 5, "min"),
            "section14_ratio": plot_section14_ratio(hist)}
    written = []
    for name, fig in figs.items():
        if fig is None:
            continue
        path = os.path.join(outdir, name + ".png")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    # §11 3D grids are rotating GIFs (static images are hard to read in 3D)
    for name, saver in (("section11_grids", save_section11_grids_gif),
                        ("section11_evolution", save_section11_evolution_gif)):
        try:
            p = saver(hist, os.path.join(outdir, name + ".gif"))
            if p:
                written.append(p)
        except Exception as e:
            print(f"  [{name}] GIF failed: {e}")
    return written
