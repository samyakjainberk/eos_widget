"""
Re-render saved runs — re-plot every panel from an existing results.pt, no re-training.

Two jobs:
  1. Migrate the function-Hessian eigenvalue series (§1 H_edge_max/min, §2 H_top, §3 H_bot) to the
     per-sample scale by dividing by N.  H = ∇²(Σf) is summed over the M=N·oc outputs, so its raw
     eigenvalues grow ∝N; the loss Hessian / sharpness average that sum by 1/N, so the two only live
     on a common scale once H is divided by N.  Idempotent: guarded by meta['hess_per_sample'].
  2. Re-run plots.save_panels with the current (fixed) plotting code.

    python -m eos_lab.rerender                       # migrate + re-render runs/*
    python -m eos_lab.rerender --runs-root some/dir  # a different sweep folder
    python -m eos_lab.rerender --no-normalize        # only re-plot, leave results.pt untouched
"""
import argparse
import glob
import math
import os

import torch

from . import plots

_H_TRACK_KEYS = ("H_top", "H_bot")          # list-of-tracks (each a list over time)
_H_SCALAR_KEYS = ("H_edge_max", "H_edge_min")  # one list over time


def _scale(x, inv_n):
    return x * inv_n if (x is not None and x == x) else x


def normalize_hessian(res):
    """Divide the stored function-Hessian eigenvalues by N, in series + history. Idempotent."""
    meta = res["meta"]
    if meta.get("hess_per_sample"):
        return False
    N = meta["nsamp"]
    inv = 1.0 / N
    s = res["series"]
    for k in _H_TRACK_KEYS:
        if k in s:
            s[k] = [[_scale(v, inv) for v in track] for track in s[k]]
    for k in _H_SCALAR_KEYS:
        if k in s:
            s[k] = [_scale(v, inv) for v in s[k]]
    for rec in res.get("history", []):     # keep per-step records consistent too
        for k in _H_TRACK_KEYS:
            if rec.get(k) is not None:
                rec[k] = [_scale(v, inv) for v in rec[k]]
        for k in _H_SCALAR_KEYS:
            if k in rec:
                rec[k] = _scale(rec[k], inv)
    meta["hess_per_sample"] = True
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Re-render saved EoS/PS runs from results.pt.")
    ap.add_argument("--runs-root", default=os.path.join(os.path.dirname(__file__), "runs"))
    ap.add_argument("--no-normalize", action="store_true",
                    help="do not migrate H eigenvalues to the per-sample scale (only re-plot)")
    a = ap.parse_args(argv)

    dirs = sorted(d for d in glob.glob(os.path.join(a.runs_root, "*")) if os.path.isdir(d))
    n_done = n_migrated = 0
    for d in dirs:
        f = os.path.join(d, "results.pt")
        if not os.path.exists(f):
            continue
        try:                                   # a job may be mid-save (torch.save isn't atomic)
            res = torch.load(f, weights_only=False)
        except Exception as e:
            print(f"  {os.path.basename(d):60s} SKIP (unreadable: {e})")
            continue
        migrated = (not a.no_normalize) and normalize_hessian(res)
        if migrated:
            torch.save(res, f)
            n_migrated += 1
        files = plots.save_panels(res, d)
        n_done += 1
        tag = " [H/N migrated]" if migrated else ""
        print(f"  {os.path.basename(d):60s} {len(files):2d} panels{tag}")
    print(f"re-rendered {n_done} runs ({n_migrated} migrated to per-sample H scale)")


if __name__ == "__main__":
    main()
