#!/usr/bin/env python
"""Headless capture runner for the PREDICTION widget (eos_widget_prediction/index_prediction.html).

Runs — offline, no browser — the SAME compute the prediction widget streams from its TWO SSE
connections, and merges both into ONE self-contained capture file you load later with the widget's
"⬆ load run":

  1. the live prediction run  — server.run_stream with the prediction widget's own section toggles ON
                                (base §1/§2/§3 + §5-Jproj + §4-SLQ + §6/§7 + Prediction-3/4/5, and — for
                                 a "capture everything" run — Prediction-6, the 4.2 ray, and the
                                 early-dynamics stats too), producing meta / step / slq / g13/g14/g16/g17
                                 / g21 / g_pred* / g_ray / g_eds / done records.
  2. the Prediction-1&2 sweep — server.run_sweep (mode=sweep): N (lr, init-std) pairs → sweeppt +
                                 (with the 2c/2d toggle) sweeppt2c scatter points.

Load the merged file in the widget and every panel — including the Prediction-1&2 sweep scatter —
replays with no server running. This is the convenient path: fan many runs across GPUs / SLURM, then
analyse them all in the browser afterwards.

  # one capture, defaults (small multi-sample MLP so every prediction panel produces):
  python eos_prediction.py --out runs_captured/pred_demo.json

  # a real config (device + more steps); predictions need N·outD ≤ grid3dcap (500 here):
  python eos_prediction.py --dataset chebyshev --nsamp 12 --steps 400 --device cuda:0 \
                           --out runs_captured/pred_cheb.json

  # skip the sweep (predictions only), or tune it:
  python eos_prediction.py --no-sweep --out runs_captured/pred_only.json
  python eos_prediction.py --swpairs 150 --swsteps 120 --out runs_captured/pred_bigsweep.json

The sweep's ‖J·ṙ‖−‖J̇·r‖ theory is squared-loss only, so run_sweep is MSE-only; a CE run captures the
predictions and simply notes that the sweep was skipped (see eos_prediction_multiclass.py).
"""
import argparse
import gzip
import json
import os
import sys
import time

import torch

import server
import capture_run


# Sections the prediction widget actually computes (mirrors index_prediction.html currentQueryParams()):
#   base panels it shows …                       … plus its own prediction sections.
# STRICT SEPARATION: every OTHER main-widget s-flag is forced OFF, exactly as the widget does, so heavy
# per-sample-Hessian sections can neither leak plots nor slow the run.
BASE_ON = ["s1", "s2", "s3", "s4", "s5", "s12", "s14", "s16", "s17", "s29", "s35"]
# s36/s37/s38 = Prediction-3/4/5 (always on in the widget). s39/s41/s42 = Prediction-6 / 4.2-ray /
# early-dynamics stats; "ss" = self-stabilization — opt-in in the widget; ON here so a capture records EVERYTHING. Drop any with --set sNN=0 / --set ss=0.
PRED_ON = ["s36", "s37", "s38", "s39", "s41", "s42", "ss"]

# Prediction-widget knob defaults (the `||'…'` fallbacks in currentQueryParams / _runSweep).
PRED_KNOBS = {
    "p4t0": "40", "p4s": "10", "p3k": "10", "p3rep": "100", "p4thr": "0.8", "p4rep": "100",
    "p5t0": "10", "stat_init": "0", "cubicapprox": "25",           # Prediction-3/4/5
    "prthr": "0.9", "prm": "40", "prK": "75", "prdir": "0.9",      # Prediction-4.2 ray (s41)
    "edsmrk": "50", "edsangk": "10",                               # early-dynamics stats (s42)
    "ssk": "5",                                                    # self-stabilization (ss): # top/bottom eigenvectors of the preconditioned Hessian H_P
    "lr_gd": "0.1", "lr_sign": "0.001", "lr_muon": "0.01", "lr_gn": "0.01",   # Prediction-6 per-optimizer LRs
}

# Prediction-1&2 sweep knob defaults (mirrors _runSweep()). The sweep uses its OWN small per-pair step
# budget (SWEEP_STEPS), NOT the main run's, so it stays fast.
SWEEP_KNOBS = {
    "swpairs": "100", "swK": "20", "swsumn": "10", "swsumn2": "10", "swmrk": "10",
    "swps5n": "4", "swcsmin": "0.1", "swcsmax": "10", "sw2cd": "1", "sw2e": "1", "swTstart": "0",
}
SWEEP_STEPS = "100"


def prediction_defaults():
    """The faithful prediction-run param dict: base model/data/cadence defaults (from capture_run) with
    ONLY the prediction widget's sections enabled + its prediction knobs."""
    p = capture_run.default_params()
    for i in range(1, 43):                       # STRICT SEPARATION: clear every section flag first …
        p["s%d" % i] = "0"
    for k in BASE_ON + PRED_ON:                  # … then enable exactly the prediction widget's sections.
        p[k] = "1"
    p.update(PRED_KNOBS)
    p["nsamp"] = "10"          # multi-sample: the Eq-21 / Eq-51 predictions need N>1 (the widget forces nsamp≥10)
    p["mode"] = "run"
    p["grid3dcap"] = "500"     # the explicit-Jacobian predictions need N·outD ≤ grid3dcap
    p["loss"] = "mse"          # squared loss (the sweep theory is MSE; CE variant is eos_prediction_multiclass.py)
    return p


# ── argument plumbing (a superset of capture_run's flags + the sweep controls) ──────────────────────
def build_argparser(desc):
    ap = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output capture path (.json, or .json.gz with --gzip)")
    ap.add_argument("--gzip", action="store_true", help="gzip the output (writes <out>.gz if --out lacks .gz)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float64"])
    ap.add_argument("--cifar-dir", default=None, help="dir with CIFAR-10 raw batches (for cifar10/cifar2/mnist)")
    ap.add_argument("--label", default=None, help="optional human label stored in the capture meta")
    ap.add_argument("--set", action="append", default=[], dest="sets",
                    help="override any widget param: --set key=value (repeatable, e.g. --set s41=0 --set swpairs=150)")
    ap.add_argument("--no-sweep", action="store_true", help="capture the prediction run only (skip the Prediction-1&2 sweep)")
    ap.add_argument("--swpairs", default=None, help="# (lr,std) pairs in the sweep (default %s)" % SWEEP_KNOBS["swpairs"])
    ap.add_argument("--swsteps", default=None, help="GD steps per sweep pair (default %s; small = fast)" % SWEEP_STEPS)
    # common direct flags (everything else via --set)
    for k in ["dataset", "arch", "loss", "act", "bias", "ssign", "initscheme", "optimizer"]:
        ap.add_argument("--" + k, default=None)
    for k in ["nsamp", "steps", "width", "depth", "indim", "outdim", "neig", "seed",
              "degree", "dmodel", "nlayer", "nhead", "seqlen", "eigevery", "batch"]:
        ap.add_argument("--" + k, default=None)
    for k in ["lr", "chmul", "init", "inputstd", "tgt"]:
        ap.add_argument("--" + k, default=None)
    return ap


def coerce_overrides(base, args):
    """defaults <- explicit CLI flags <- --set key=value pairs."""
    p = dict(base)
    direct = ["dataset", "arch", "loss", "act", "bias", "ssign", "initscheme", "optimizer",
              "nsamp", "steps", "width", "depth", "indim", "outdim", "neig", "seed",
              "degree", "dmodel", "nlayer", "nhead", "seqlen", "eigevery", "batch",
              "lr", "chmul", "init", "inputstd", "tgt"]
    for k in direct:
        v = getattr(args, k, None)
        if v is not None:
            p[k] = str(v)
    for kv in args.sets:
        if "=" not in kv:
            raise SystemExit("--set expects key=value, got: %r" % kv)
        k, v = kv.split("=", 1)
        p[k.strip()] = v.strip()
    return p


# ── the sweep pass ──────────────────────────────────────────────────────────────────────────────────
def capture_sweep(run_params, sweep_steps, sweep_knobs, device, dtype, cifar_dir, progress=True):
    """Run server.run_sweep once (mode=sweep) with the same model config; return ONLY the sweeppt /
    sweeppt2c data records (its own meta/done are dropped so they can't collide with the main run's
    meta). run_sweep is MSE-only: a CE run yields an error meta and no points, surfaced as `note`
    but not treated as fatal."""
    server.DTYPE = dtype
    server.DEVICE = device
    server.DEVICE_POOL = [device]
    server._TL.device = device
    server._TL.cifar_dir = cifar_dir
    if cifar_dir:
        server.CIFAR_DIR = cifar_dir
    if device.type == "cuda":
        torch.cuda.set_device(device)
    q = {k: [str(v)] for k, v in run_params.items()}
    q["mode"] = ["sweep"]
    q["steps"] = [str(sweep_steps)]
    for k, v in sweep_knobs.items():
        q[k] = [str(v)]
    P = server._parse_params(q)
    pts, note = [], None
    t0 = time.time(); last = t0
    try:
        for msg in server.run_sweep(P):
            t = msg.get("type")
            if t in ("sweeppt", "sweeppt2c"):
                pts.append(server._sanitize(msg))
            elif t == "meta" and msg.get("error"):
                note = msg["error"]
            elif t == "error":
                note = msg.get("error", "sweep error")
            now = time.time()
            if progress and (now - last) > 2.0:
                last = now
                sys.stderr.write("  … sweep %d points\n" % len(pts)); sys.stderr.flush()
    except KeyboardInterrupt:
        sys.stderr.write("  sweep interrupted — keeping %d points\n" % len(pts))
    return pts, note, time.time() - t0


def merge_sweep(run_records, sweep_pts):
    """Insert the sweep points just before the run's trailing done/error record (or append if none),
    so the merged stream still ends on the terminal record the loader expects."""
    out = list(run_records)
    idx = next((i for i in range(len(out) - 1, -1, -1)
                if out[i].get("type") in ("done", "error")), None)
    if idx is None:
        out.extend(sweep_pts)
    else:
        out[idx:idx] = sweep_pts
    return out


# ── output ────────────────────────────────────────────────────────────────────────────────────────
def write_capture(records, params, dev, dtype, label, out, use_gz, wall, final, out_path=None, compacted=False):
    """Serialize the (possibly partial) capture atomically (tmp + rename)."""
    op = out_path or out
    by_type = {}
    for r in records:
        by_type[r.get("type", "?")] = by_type.get(r.get("type", "?"), 0) + 1
    meta = next((r for r in records if r.get("type") == "meta"), {})
    obj = {
        "format": "eos-widget-capture/v1",
        "kind": "run",                        # a merged prediction+sweep capture loads via the normal run path
        "created_unix": int(time.time()),
        "label": label or "",
        "device": str(dev),
        "dtype": str(dtype).replace("torch.", ""),
        "wall_sec": round(wall, 2),
        "partial": not final,
        "compacted": compacted,
        "params": params,                     # query-style dict → restores the widget controls (incl. sweep knobs)
        "p": meta.get("p"),
        "record_counts": by_type,
        "records": records,
    }
    data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    tmp = op + ".tmp"
    opener = gzip.open if use_gz else open
    with opener(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, op)
    return by_type, len(data)


def run_capture(defaults_fn, argv=None, title="prediction"):
    """Full flow: prediction run + sweep → merged capture. `defaults_fn` supplies the base param dict
    (prediction_defaults, or the multiclass variant)."""
    ap = build_argparser(__doc__ if title == "prediction" else "Headless capture for the %s widget." % title)
    a = ap.parse_args(argv)

    # device + dtype
    if a.device == "auto":
        dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        dev = torch.device(a.device)
    if a.dtype == "auto":
        dtype = torch.float32 if dev.type == "cuda" else torch.float64
    else:
        dtype = torch.float32 if a.dtype == "float32" else torch.float64

    params = coerce_overrides(defaults_fn(), a)
    # also record the sweep knobs in the saved params so the widget restores the sweep controls too
    sweep_knobs = dict(SWEEP_KNOBS)
    if a.swpairs is not None:
        sweep_knobs["swpairs"] = str(a.swpairs)
    sweep_steps = str(a.swsteps) if a.swsteps is not None else SWEEP_STEPS
    for k, v in sweep_knobs.items():
        params.setdefault(k, v)

    out = a.out
    use_gz = a.gzip or out.endswith(".gz")
    if a.gzip and not out.endswith(".gz"):
        out = out + ".gz"
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    print("[%s] device=%s dtype=%s dataset=%s arch=%s loss=%s nsamp=%s steps=%s sweep=%s → %s"
          % (title, dev, str(dtype).replace("torch.", ""), params["dataset"], params["arch"],
             params.get("loss", "mse"), params["nsamp"], params["steps"],
             ("off" if a.no_sweep else sweep_knobs["swpairs"] + "×" + sweep_steps), out), flush=True)

    t_start = time.time()

    # ---- 1) the prediction run (reuses capture_run's device setup + partial-save machinery) ----
    def _save_partial(recs):
        write_capture(recs, params, dev, dtype, a.label, out, use_gz, time.time() - t_start, final=False)

    run_records, run_wall = capture_run.capture(params, dev, dtype, a.cifar_dir, partial_save=_save_partial)
    print("[%s] prediction run: %d records (%.1fs)" % (title, len(run_records), run_wall), flush=True)

    # ---- 2) the Prediction-1&2 sweep ----
    records = run_records
    sweep_note = None
    if not a.no_sweep:
        sweep_pts, sweep_note, sweep_wall = capture_sweep(params, sweep_steps, sweep_knobs, dev, dtype, a.cifar_dir)
        if sweep_pts:
            records = merge_sweep(run_records, sweep_pts)
            print("[%s] sweep: %d points (%.1fs) merged" % (title, len(sweep_pts), sweep_wall), flush=True)
        else:
            print("[%s] sweep: 0 points (%.1fs)%s" % (title, sweep_wall,
                  (" — " + sweep_note) if sweep_note else ""), flush=True)

    wall = time.time() - t_start
    by_type, nbytes = write_capture(records, params, dev, dtype, a.label, out, use_gz, wall, final=True)
    print("[%s] wrote %s  (%d records, %.1f KB, %.1fs)" % (title, out, len(records), nbytes / 1024.0, wall), flush=True)
    print("[%s] record types: %s" % (title, ", ".join("%s×%d" % (k, v) for k, v in sorted(by_type.items()))), flush=True)

    # compacted <out>.min.json (the §14 N³ cubes reduced to the time-series stats the browser plots) — load this one
    try:
        import compact_capture as _cc
        n14 = sum(_cc.compact_record(r) for r in records)
        if out.endswith(".json.gz"):
            min_out = out[:-8] + ".min.json.gz"
        elif out.endswith(".json"):
            min_out = out[:-5] + ".min.json"
        else:
            min_out = out + ".min"
        _bt, mbytes = write_capture(records, params, dev, dtype, a.label, out, use_gz, wall, final=True,
                                    out_path=min_out, compacted=True)
        print("[%s] also wrote %s  (compacted: %d g14 reduced, %.1f KB, %.1fx smaller) — load this one"
              % (title, min_out, n14, mbytes / 1024.0, nbytes / max(mbytes, 1)), flush=True)
    except Exception as e:
        print("[%s] compaction step skipped (%s) — full capture is fine, just larger" % (title, e), flush=True)

    if "error" in by_type or any(r.get("type") == "meta" and r.get("error") for r in records):
        print("[%s] WARNING: run reported an error record — check the config." % title, flush=True)


def main():
    run_capture(prediction_defaults, title="prediction")


if __name__ == "__main__":
    main()
