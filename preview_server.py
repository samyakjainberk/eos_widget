#!/usr/bin/env python3
"""Headless preview of the *server-backed* widget (index_server.html + server.py).

index_server.html does no math of its own — its JS numeric core is dead code ("kept
verbatim from index.html for parity"). At runtime the page streams per-step diagnostics
from server.py over Server-Sent Events and only draws them with Plotly.

This script is the Python equivalent of that page: it drives server.py's REAL
`run_stream()` (the exact code path the browser hits at `/run`), replays the SSE
messages the way the browser's `pushPoint` / `refreshSLQ` do, and renders the six
panel-sections to PNGs — so you can confirm the server build is fine end to end
without starting the server, a browser, or an SSH tunnel.

    python3 preview_server.py                       # all 3 presets, float64, auto device
    python3 preview_server.py --preset product
    python3 preview_server.py --device cpu --dtype float64
    python3 preview_server.py --device cuda:0 --dtype float32   # exactly what the live widget runs

Output: preview_server_<preset>.png  (named so they sit next to the index.html-path
previews preview_<preset>.png for an at-a-glance comparison).
"""
import argparse
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import server  # the actual GPU backend module — we call its run_stream() directly

PAL = ['#2563eb', '#d35400', '#16a34a', '#9333ea', '#dc2626', '#0891b2', '#ea580c', '#475569']

# Preset controls — identical to the PRESETS object in index_server.html. bias/fixedx/act
# stay strings because run_stream compares them (P["bias"] == "1", etc.). All six panel
# toggles default ON, matching the checked en1..en6 checkboxes on the page.
PRESETS = {
    "product": dict(depth=5, width=1,  act="linear", bias="0", fixedx="1", inputstd=1.0, lr=0.36, init=0.4, nsamp=1,  indim=1, outdim=1, tgt=1.0, neig=3, kth=1, steps=400, eigevery=1, slqprobes=4, energyp=99, gson="1", seed=0),
    "simple":  dict(depth=3, width=20, act="tanh",   bias="1", fixedx="1", inputstd=1.0, lr=0.16, init=0.3, nsamp=1,  indim=1, outdim=1, tgt=5.0, neig=3, kth=1, steps=120, eigevery=2, slqprobes=4, energyp=99, gson="1", seed=0),
    "multi":   dict(depth=3, width=3,  act="tanh",   bias="1", fixedx="0", inputstd=1.0, lr=0.35, init=1.2, nsamp=40, indim=4, outdim=1, tgt=1.0, neig=3, kth=1, steps=500, eigevery=2, slqprobes=4, energyp=99, gson="1", seed=11),
}
TITLES = {
    "product": "Single-sample deep-linear product — sustained EoS  [seed 0]",
    "simple":  "Single-sample tanh MLP — progressive sharpening, no EoS  [seed 0]",
    "multi":   "Multi-sample tanh MLP — PS -> EoS oscillation  [seed 11]",
}


def make_params(preset):
    """Shape a preset into the dict run_stream() consumes (mirrors backendURL + _parse_params)."""
    P = dict(preset)
    P.update(s1=True, s2=True, s3=True, s4=True, s5=True, s6=True,
             gs=(str(preset["gson"]) == "1"), start=0)
    return P


def _nan(v):
    return np.nan if v is None else float(v)


def collect(P):
    """Consume server.run_stream(P) and assemble per-step series, exactly as the browser
    accumulates step messages and keeps only the latest SLQ density per matrix."""
    D = {k: [] for k in ("T", "L", "SH", "R", "HfMax", "HfMin",
                          "HfT", "HfB", "HLT", "HLB", "GNT", "GNB", "SRT", "SRB",
                          "paPosMx", "paPosMn", "paNegMx", "paNegMn", "dimPos", "dimNeg")}
    JT = JB = None
    slq = {}
    meta = None

    for m in server.run_stream(P):
        kind = m["type"]
        if kind == "meta":
            if m.get("error"):
                raise RuntimeError("server: " + m["error"])
            meta = m
        elif kind == "step":
            D["T"].append(m["t"])
            D["L"].append(_nan(m["loss"]))
            D["SH"].append(_nan(m["sharp"]))
            D["R"].append(np.array(m["r"], dtype=float))
            D["HfMax"].append(_nan(m["hfMax"]))
            D["HfMin"].append(_nan(m["hfMin"]))
            D["HfT"].append(m["hfTop"]); D["HfB"].append(m["hfBot"])
            D["HLT"].append(m["hlTop"]); D["HLB"].append(m["hlBot"])
            D["GNT"].append(m["gnTop"]); D["GNB"].append(m["gnBot"])
            D["SRT"].append(m["srTop"]); D["SRB"].append(m["srBot"])
            pp, pn = m["paPos"], m["paNeg"]
            D["paPosMx"].append(pp["mx"] if pp else np.nan)
            D["paPosMn"].append(pp["mn"] if pp else np.nan)
            D["paNegMx"].append(pn["mx"] if pn else np.nan)
            D["paNegMn"].append(pn["mn"] if pn else np.nan)
            D["dimPos"].append(m["dimPos"]); D["dimNeg"].append(m["dimNeg"])
            jt, jb = m["jt"], m["jb"]
            if JT is None:
                JT = [[] for _ in jt]; JB = [[] for _ in jb]
            for i, v in enumerate(jt): JT[i].append(_nan(v))
            for i, v in enumerate(jb): JB[i].append(_nan(v))
        elif kind == "slq":
            for key in ("sH", "sG", "sS", "sHL"):
                if m.get(key):
                    slq[key] = (np.array(m[key]["x"]), np.array(m[key]["y"]))
        elif kind == "error":
            raise RuntimeError("server: " + m["error"])
        elif kind == "done":
            break

    if meta is None or not D["T"]:
        raise RuntimeError("no data streamed (diverged immediately?)")

    # GN/SR are None at every step when the "G & S spectra" toggle is off → keep them None
    # (the panel draws an "(off)" placeholder); otherwise stack into a (steps, n) array.
    for k in ("HfT", "HfB", "HLT", "HLB", "GNT", "GNB", "SRT", "SRB"):
        D[k] = None if any(row is None for row in D[k]) else np.array(D[k], dtype=float)
    D["JT"] = JT or []
    D["JB"] = JB or []
    D["slq"] = slq
    D.update(n=meta["n"], kth=meta["kth"], nResid=meta["nResid"],
             thr=meta["thr"], p=meta["p"], device=meta["device"])
    return D


# ----- rendering (mirrors the six sections of index_server.html / _preview.py) -----
def add_thr(ax, thr, data):
    smax = np.nanmax(data) if np.isfinite(np.nanmax(data)) else 1e-9
    if thr <= 1.5 * max(smax, 1e-9):
        ax.axhline(thr, ls="--", c="#9ca3af", lw=1.2, label="2/lr"); ax.legend(fontsize=7)
    else:
        ax.text(0.98, 0.96, f"2/lr={thr:.0f} (off scale)", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#6b7280")


def lines(ax, x, Mat, n, names=None):
    if Mat is None:  # quantity not computed (G & S toggle off)
        ax.text(0.5, 0.5, "(off)", transform=ax.transAxes, ha="center", va="center", color="#9ca3af")
        return
    for i in range(n):
        ax.plot(x, Mat[:, i], color=PAL[i % len(PAL)], lw=1.5,
                label=(names[i] if names else None))


def slqax(ax, dens, c, title):
    if not dens:
        ax.text(0.5, 0.5, "(no SLQ)", transform=ax.transAxes, ha="center", va="center", color="#9ca3af")
        ax.set_title(title); return
    x, y = dens
    ax.fill_between(x, 0, y, color=c, alpha=0.18); ax.plot(x, y, color=c, lw=1.6)
    ax.set_title(title); ax.set_xlabel("eigenvalue")


def render(D, fname, title):
    x = D["T"]; n = D["n"]; kth = D["kth"]; thr = D["thr"]
    fig = plt.figure(figsize=(21, 25)); fig.suptitle(title, fontsize=14, y=0.998)
    gs = fig.add_gridspec(6, 4, hspace=0.5, wspace=0.42)
    # §1 — loss / sharpness / residual / H extreme eigs
    ax = fig.add_subplot(gs[0, 0]); ax.semilogy(x, D["L"], color="#2563eb"); ax.set_title("loss"); ax.set_xlabel("GD step")
    ax = fig.add_subplot(gs[0, 1]); ax.plot(x, D["SH"], color="#dc2626"); add_thr(ax, thr, D["SH"]); ax.set_title("sharpness  λmax(∇²loss)"); ax.set_xlabel("GD step")
    ax = fig.add_subplot(gs[0, 2]); R = np.array(D["R"])
    for j in range(D["nResid"]): ax.plot(x, R[:, j], lw=1.2, color=PAL[j % len(PAL)])
    ax.axhline(0, color="#d1d5db", lw=1); ax.set_title("residual  y − f(x)"); ax.set_xlabel("GD step")
    ax = fig.add_subplot(gs[0, 3]); ax.plot(x, D["HfMax"], color="#2563eb", label="λmax"); ax.plot(x, D["HfMin"], color="#dc2626", label="λmin")
    ax.axhline(0, color="#d1d5db", lw=1); ax.set_title("function Hessian H — largest & smallest eig"); ax.set_xlabel("GD step"); ax.legend(fontsize=7)
    # §2 — top-n
    for col, (key, ttl, has_thr) in enumerate([("HfT", "function Hessian H — top-n", 0), ("HLT", "loss Hessian — top-n", 1), ("GNT", "Gauss–Newton G — top-n", 0), ("SRT", "residual term S — top-n", 0)]):
        ax = fig.add_subplot(gs[1, col]); lines(ax, x, D[key], n)
        if has_thr: add_thr(ax, thr, D[key][:, 0])
        ax.set_title(ttl); ax.set_xlabel("GD step")
    # §3 — bottom-n
    for col, (key, ttl) in enumerate([("HfB", "function Hessian H — bottom-n"), ("HLB", "loss Hessian — bottom-n"), ("GNB", "Gauss–Newton G — bottom-n"), ("SRB", "residual term S — bottom-n")]):
        ax = fig.add_subplot(gs[2, col]); lines(ax, x, D[key], n); ax.axhline(0, color="#d1d5db", lw=1); ax.set_title(ttl); ax.set_xlabel("GD step")
    # §5 — SLQ spectra
    slqax(fig.add_subplot(gs[3, 0]), D["slq"].get("sH"), "#2563eb", "SLQ — function Hessian H")
    slqax(fig.add_subplot(gs[3, 1]), D["slq"].get("sG"), "#16a34a", "SLQ — Gauss–Newton G")
    slqax(fig.add_subplot(gs[3, 2]), D["slq"].get("sS"), "#d35400", "SLQ — residual term S")
    slqax(fig.add_subplot(gs[3, 3]), D["slq"].get("sHL"), "#dc2626", "SLQ — full loss Hessian")
    # §6 — eigenspace rotation
    ax = fig.add_subplot(gs[4, 0:2]); ax.plot(x, D["paPosMx"], color="#2563eb", label="max"); ax.plot(x, D["paPosMn"], color="#94a3b8", ls=":", label="mean"); ax.set_title("positive eigenspace — principal angle (deg)"); ax.set_xlabel("GD step"); ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[4, 2:4]); ax.plot(x, D["paNegMx"], color="#2563eb", label="max"); ax.plot(x, D["paNegMn"], color="#94a3b8", ls=":", label="mean"); ax.set_title("negative eigenspace — principal angle (deg)"); ax.set_xlabel("GD step"); ax.legend(fontsize=7)
    # §4 — J projections
    ax = fig.add_subplot(gs[5, 0])
    for i in range(n): ax.plot(x, D["JT"][i], color=PAL[i % len(PAL)], label=f"top-{i+1}")
    ax.set_title(f"J onto top-{n} eigvecs"); ax.set_xlabel("GD step"); ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[5, 1])
    for i in range(n): ax.plot(x, D["JB"][i], color=PAL[i % len(PAL)], label=f"bot-{i+1}")
    ax.set_title(f"J onto bottom-{n} eigvecs"); ax.set_xlabel("GD step"); ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[5, 2]); ax.plot(x, D["JT"][kth - 1], color="#2563eb"); ax.set_title(f"J onto the {kth}-th TOP eigvec"); ax.set_xlabel("GD step")
    ax = fig.add_subplot(gs[5, 3]); ax.plot(x, D["JB"][kth - 1], color="#dc2626"); ax.set_title(f"J onto the {kth}-th BOTTOM eigvec"); ax.set_xlabel("GD step")
    fig.savefig(fname, dpi=90, bbox_inches="tight"); plt.close(fig)


def run_preset(name):
    P = make_params(PRESETS[name])
    D = collect(P)
    fname = f"preview_server_{name}.png"
    render(D, fname, "Preset: " + TITLES[name])
    sh = np.array(D["SH"], dtype=float)
    print(f"{fname}: device={D['device']} p={D['p']} 2/lr={D['thr']:.2f} "
          f"sharp {sh[min(1, len(sh)-1)]:.2f}->{np.nanmax(sh):.2f} "
          f"| H_f [{D['HfMin'][-1]:.2f},{D['HfMax'][-1]:.2f}] "
          f"dim+={D['dimPos'][-1]} dim-={D['dimNeg'][-1]} "
          f"| max-PA+={np.nanmax(D['paPosMx']):.1f}deg")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS) + ["all"], default="all")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"],
                    help="float64 (default) matches the browser's double precision; "
                         "float32 matches the live GPU widget")
    a = ap.parse_args()

    server.DEVICE = server.pick_device(a.device)
    server.DTYPE = torch.float64 if a.dtype == "float64" else torch.float32
    torch.set_grad_enabled(False)
    if server.DEVICE.type == "cuda":
        torch.cuda.set_device(server.DEVICE)

    print(f"driving server.run_stream on {server.DEVICE} · {str(server.DTYPE).replace('torch.', '')}")
    names = list(PRESETS) if a.preset == "all" else [a.preset]
    for nm in names:
        run_preset(nm)
    print("done.")


if __name__ == "__main__":
    main()
