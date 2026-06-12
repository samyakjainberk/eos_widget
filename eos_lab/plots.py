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
    if "resid_head" in series:              # per-sample residual y−f (first nResid) — matches the widget's panel
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


def save_panels(results, outdir):
    """Render every supported section to PNGs in `outdir`. Returns the list of files written."""
    os.makedirs(outdir, exist_ok=True)
    series, meta, hist = results["series"], results["meta"], results["history"]
    figs = {"section1_loss_sharpness": plot_section1(series, meta),
            "section2_top_eig": plot_section2(series),
            "section3_bottom_eig": plot_section3(series),
            "section4_J_onto_H": plot_section4(series),
            "section5_slq": plot_slq(hist, meta.get("nsamp", 1)),
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
            "section9dc_selfresidual_vs_full_hessian": plot_section9dc(series, meta)}
    written = []
    for name, fig in figs.items():
        if fig is None:
            continue
        path = os.path.join(outdir, name + ".png")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written
