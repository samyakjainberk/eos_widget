#!/usr/bin/env python
"""Headless capture runner for the EoS / Progressive-Sharpening widget.

Runs the SAME compute as the GPU backend's /run SSE stream (server.run_stream — sections §1–§17,
i.e. every en1…en25 toggle) but, instead of streaming to a live browser, CAPTURES every record to a
single self-contained file you can load into the widget later (Compute → "Load saved run").

This lets you fan many runs out across SLURM / multiple GPUs and analyse them all in the browser
afterwards, without keeping a server (or the tab) open while they compute.

  # one capture, all sections on, defaults (small multi-sample MLP so every section actually produces):
  python capture_run.py --out runs_captured/demo.json

  # a real config (overrides are plain widget params; section toggles default ON):
  python capture_run.py --dataset cifar10 --arch vgg11 --nsamp 25 --steps 400 \
                        --device cuda:0 --out runs_captured/cifar_vgg.json

  # disable a couple of heavy sections, keep the rest:
  python capture_run.py --set s17=0 --set s19=0 --out runs_captured/light.json

The output file is JSON (add --gzip for .json.gz). Load it from the widget's Compute dropdown.

Every section toggle (s1…s24, s25 and the §11–§17 advanced ones s18…s23) defaults to ON, and so do
the §16/§17 baselines (s24base/s25base), so "everything in the toggle list" is run by default. Whether
a section actually produces records still depends on its gates (multi-sample / small-N / loss type) —
exactly as in the live widget; the browser shows a "not produced" note for any gated-out section.
"""
import argparse
import gzip
import json
import os
import sys
import time

import torch

import server


# Every widget parameter the browser can send, with the SAME defaults as index.html / _parse_params,
# EXCEPT every section toggle (s*) and surrogate panel toggle (c*) defaults to "1" (ON) so a capture
# records the full toggle list unless the user turns something off.
def default_params():
    p = {
        # ---- model / data ----
        "depth": "2", "width": "8", "act": "tanh", "bias": "1", "fixedx": "0",
        "inputstd": "1.0", "ssign": "off", "optimizer": "gd", "lr": "0.05", "init": "0.5",
        "initscheme": "default", "nsamp": "8", "batch": "0", "indim": "4", "outdim": "3",
        "tgt": "1.0", "dataset": "synthetic", "arch": "mlp", "loss": "mse",
        "chmul": "0.25", "nlayer": "2", "nhead": "4", "dmodel": "64", "seqlen": "16",
        "vocab": "50257", "degree": "3", "cvar": "0.0", "divreg": "0.3",
        "c2a": "0", "c2b": "1",
        # ---- diagnostics cadence / sizes ----
        "neig": "4", "kth": "1", "steps": "120", "eigevery": "2", "heavyevery": "4",
        "slqprobes": "4", "energyp": "99", "seed": "0", "start": "0",
        "qapprox": "25", "qmode": "1", "tset": "3", "cubicapprox": "10",
        "grid3dcap": "500", "sec12ncap": "500",
        "s24warm": "5", "s24iter": "120", "s24grid": "0.1", "s24ares": "0.01", "s24k": "32",
        "gson": "1", "mode": "run", "surrogate": "quad",
    }
    # main-run section toggles s1..s23 + s24(§16) + s25(§17), ALL ON
    for i in list(range(1, 24)) + [24, 25]:
        p["s%d" % i] = "1"
    p["s24base"] = "1"   # §16 five baselines
    p["s25base"] = "1"   # §17 five baselines
    # surrogate-compare panel toggles c1..c12, ALL ON (used only if a surrogate capture is requested)
    for i in range(1, 13):
        p["c%d" % i] = "1"
    return p


def coerce_overrides(args, extra_sets):
    """Build the param dict: defaults <- explicit CLI flags <- --set key=value pairs."""
    p = default_params()
    # explicit, commonly-used flags
    direct = ["dataset", "arch", "loss", "nsamp", "steps", "lr", "width", "depth", "act",
              "bias", "indim", "outdim", "neig", "seed", "degree", "chmul", "dmodel",
              "nlayer", "nhead", "seqlen", "init", "initscheme", "eigevery", "batch",
              "ssign", "inputstd", "tgt", "optimizer", "s24iter", "s24warm", "s24k"]
    for k in direct:
        v = getattr(args, k, None)
        if v is not None:
            p[k] = str(v)
    # arbitrary key=value overrides (e.g. --set s17=0 --set grid3dcap=64)
    for kv in extra_sets:
        if "=" not in kv:
            raise SystemExit("--set expects key=value, got: %r" % kv)
        k, v = kv.split("=", 1)
        p[k.strip()] = v.strip()
    return p


def capture(params, device, dtype, cifar_dir, progress=True):
    """Run server.run_stream once with `params` and return the list of sanitized records."""
    server.DTYPE = dtype
    server.DEVICE = device
    server.DEVICE_POOL = [device]
    server._TL.device = device
    server._TL.cifar_dir = cifar_dir
    if cifar_dir:
        server.CIFAR_DIR = cifar_dir
    if device.type == "cuda":
        torch.cuda.set_device(device)
    # token bookkeeping so run_stream's "newer run on this device" abort never fires
    tok = server.RUN_TOKEN.get(str(device), 0) + 1
    server.RUN_TOKEN[str(device)] = tok

    q = {k: [str(v)] for k, v in params.items()}
    P = server._parse_params(q)
    P["_token"] = tok

    records = []
    t0 = time.time()
    gen = server.run_surrogate_compare(P) if P.get("mode") == "surrogate" else server.run_stream(P)
    last = t0
    for msg in gen:
        records.append(server._sanitize(msg))
        if progress and (time.time() - last) > 2.0:
            last = time.time()
            kinds = records[-1].get("type", "?")
            sys.stderr.write("  … %d records  (latest: %s, %.0fs)\n" % (len(records), kinds, time.time() - t0))
            sys.stderr.flush()
    return records, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output capture path (.json, or .json.gz with --gzip)")
    ap.add_argument("--gzip", action="store_true", help="gzip the output (writes <out>.gz if --out lacks .gz)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float64"])
    ap.add_argument("--cifar-dir", default=None, help="dir with CIFAR-10 raw batches (for cifar10/cifar2)")
    ap.add_argument("--label", default=None, help="optional human label stored in the capture meta")
    ap.add_argument("--set", action="append", default=[], dest="sets",
                    help="override any widget param: --set key=value (repeatable)")
    # common direct flags (everything else via --set)
    for k in ["dataset", "arch", "loss", "act", "bias", "ssign", "initscheme", "optimizer"]:
        ap.add_argument("--" + k, default=None)
    for k in ["nsamp", "steps", "width", "depth", "indim", "outdim", "neig", "seed",
              "degree", "dmodel", "nlayer", "nhead", "seqlen", "eigevery", "batch",
              "s24iter", "s24warm", "s24k"]:
        ap.add_argument("--" + k, default=None)
    for k in ["lr", "chmul", "init", "inputstd", "tgt"]:
        ap.add_argument("--" + k, default=None)
    a = ap.parse_args()

    # device + dtype
    if a.device == "auto":
        dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        dev = torch.device(a.device)
    if a.dtype == "auto":
        dtype = torch.float32 if dev.type == "cuda" else torch.float64
    else:
        dtype = torch.float32 if a.dtype == "float32" else torch.float64

    params = coerce_overrides(a, a.sets)

    print("[capture] device=%s dtype=%s dataset=%s arch=%s nsamp=%s steps=%s"
          % (dev, str(dtype).replace("torch.", ""), params["dataset"], params["arch"],
             params["nsamp"], params["steps"]), flush=True)
    records, wall = capture(params, dev, dtype, a.cifar_dir)

    # summary of what was produced (which record types appeared)
    by_type = {}
    for r in records:
        by_type[r.get("type", "?")] = by_type.get(r.get("type", "?"), 0) + 1
    meta = next((r for r in records if r.get("type") == "meta"), {})
    obj = {
        "format": "eos-widget-capture/v1",
        "kind": params.get("mode", "run"),
        "created_unix": int(time.time()),
        "label": a.label or "",
        "device": str(dev),
        "dtype": str(dtype).replace("torch.", ""),
        "wall_sec": round(wall, 2),
        "params": params,                 # query-style param dict → restores the widget controls
        "p": meta.get("p"),
        "record_counts": by_type,
        "records": records,
    }

    out = a.out
    use_gz = a.gzip or out.endswith(".gz")
    if a.gzip and not out.endswith(".gz"):
        out = out + ".gz"
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    if use_gz:
        with gzip.open(out, "wb") as f:
            f.write(data)
    else:
        with open(out, "wb") as f:
            f.write(data)

    print("[capture] wrote %s  (%d records, %.1f KB, %.1fs)" %
          (out, len(records), len(data) / 1024.0, wall), flush=True)
    print("[capture] record types: " + ", ".join("%s×%d" % (k, v) for k, v in sorted(by_type.items())), flush=True)
    if "error" in by_type or any(r.get("type") == "meta" and r.get("error") for r in records):
        print("[capture] WARNING: run reported an error record — check the config.", flush=True)


if __name__ == "__main__":
    main()
