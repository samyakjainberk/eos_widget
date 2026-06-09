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
    if "resid_rms" in series:
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
    """§7a — NTK alignment (always-on panel): residual onto top-n NTK eigenvectors, and the top NTK
    eigenvector onto the right-singular vectors of the function-Hessian tensor."""
    return _tracks_panel(series, [("ntkR", "residual·NTK eigvec"), ("ntkH", "NTK u₁·FH y_k")],
                         "§7a — NTK alignment (residual→NTK, NTK→FH-SVD)")


def plot_section7(series):
    return _tracks_panel(series, [("fhEvT", "FH eig (top)"), ("jhe1", "U_J·FH-vec₁"),
                                  ("jh2e1", "U_J·FH-vec₂")],
                         "§7 — multi-sample NTK & function-Hessian tensor SVD")


def plot_section8(series):
    return _tracks_panel(series, [("g2Jn", "vec(J)·G2 singvec /‖J‖_F"), ("g2J", "vec(J)·G2 singvec")],
                         "§8 — vec(J) onto FH-reshape singular vectors")


def plot_section4b(series):
    return _tracks_panel(series, [("q9tN", "J·r→Q[u₁]top /‖Jr‖"), ("q9bN", "J·r→Q[u₁]bot /‖Jr‖"),
                                  ("q9gt", "g₁·Q[u₁]top")],
                         "§4b — J·r onto eigenvectors of Q[u₁]")


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


def plot_section9(series, meta):
    """§9 — theory (Eq-13/21/27/29) vs empirical σ₁. Only the columns with data are drawn:
    Eq-13 is single-sample (M==1) so it is empty for multi-output datasets, and the whole panel
    is skipped when neither theory nor empirical produced any finite point (e.g. multi_ok=False)."""
    thP = series.get("thP"); thA = series.get("thA"); thPpsd = series.get("thPpsd")
    if thP is None and thA is None:
        return None
    t = series["t"]
    labels = ["Eq-13 (single)", "Eq-21 (col-2)", "Eq-27 (col-3)", "Eq-29 (col-4)"]

    def col(arr, i):
        return arr[i] if (arr is not None and i < len(arr)) else None

    # keep only columns that have at least one finite point in predicted or empirical
    present = [i for i in range(4)
               if any(y == y for y in (col(thP, i) or [])) or any(y == y for y in (col(thA, i) or []))]
    if not present:
        return None

    fig, axs = plt.subplots(1, len(present), figsize=(4.5 * len(present), 3.3), squeeze=False)
    for ax, i in zip(axs[0], present):
        emp = [y for y in (col(thA, i) or []) if y == y]
        prd = [y for y in (col(thP, i) or []) if y == y]
        prdP = [y for y in (col(thPpsd, i) or []) if y == y]
        if col(thP, i) is not None:
            ax.plot(*_finite(t, col(thP, i)), label="predicted")
        if col(thPpsd, i) is not None:                       # §9b: prediction + 2nd-order PSD term ‖ΔJᵀu₁‖²
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
    fig.suptitle("§9 — theoretical vs empirical sharpness", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_section9c(series, meta):
    """§9c — the same Eq-13/21/27/29 σ₁ predictions, but compared against the FULL loss-Hessian
    sharpness λmax(∇²L)=λmax(G+S) (the EoS quantity) instead of the Gauss-Newton edge λmax(G).
    Each column: predicted, predicted+PSD, and the full-Hessian sharpness; the gap is the residual S."""
    thP = series.get("thP"); thPpsd = series.get("thPpsd"); thAH = series.get("thAH")
    if thP is None or thAH is None or not any(y == y for y in thAH):
        return None
    t = series["t"]
    labels = ["Eq-13 (single)", "Eq-21 (col-2)", "Eq-27 (col-3)", "Eq-29 (col-4)"]

    def col(arr, i):
        return arr[i] if (arr is not None and i < len(arr)) else None

    present = [i for i in range(4) if any(y == y for y in (col(thP, i) or []))]
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
    fig.suptitle("§9c — σ₁ predictions vs the full-Hessian sharpness", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


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
            "section9_theory_vs_empirical": plot_section9(series, meta),
            "section9c_theory_vs_full_hessian": plot_section9c(series, meta)}
    written = []
    for name, fig in figs.items():
        if fig is None:
            continue
        path = os.path.join(outdir, name + ".png")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written
