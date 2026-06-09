"""
Command-line runner — launch a job on a GPU yourself, no browser.

    python -m eos_lab.cli --preset msample --device cuda:0 --out runs/msample
    python -m eos_lab.cli --preset cifar_vgg --device cuda:0 --set chmul=0.5 --set nsamp=64
    python -m eos_lab.cli --list-presets

Saves to --out:  results.pt (full history, torch.load-able), summary.json, and section*.png panels.
MIRRORS: pressing Run in index.html, but headless and scriptable.
"""
import argparse
import dataclasses
import json
import os

from .config import Config, PRESETS
from .train import run_job
from .logging_wandb import RunLogger, run_name


def _coerce(field_type, s):
    if field_type is int:
        return int(float(s))
    if field_type is float:
        return float(s)
    return s


def _apply_overrides(cfg, sets):
    types = {f.name: f.type for f in dataclasses.fields(Config)}
    for kv in sets or []:
        if "=" not in kv:
            raise SystemExit(f"--set expects key=value, got '{kv}'")
        k, v = kv.split("=", 1)
        if k not in types:
            raise SystemExit(f"unknown config field '{k}'. Fields: {', '.join(types)}")
        setattr(cfg, k, _coerce(types[k], v))
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run an Edge-of-Stability / Progressive-Sharpening job.")
    ap.add_argument("--preset", default="linear", help=f"one of: {', '.join(PRESETS)}")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--dtype", default=None, choices=[None, "float32", "float64"])
    ap.add_argument("--steps", type=int, default=None, help="override step count")
    ap.add_argument("--out", default=None, help="output dir (default: runs/<preset>)")
    ap.add_argument("--cifar-dir", default=None, help="dir with CIFAR-10 raw batches")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--wandb", action="store_true", help="log metrics + panels to Weights & Biases")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--run-name", default=None, help="override the auto hyperparameter run name")
    ap.add_argument("--out-root", default="runs", help="parent dir; the run folder is <out-root>/<run_name>")
    ap.add_argument("--set", action="append", metavar="KEY=VAL",
                    help="override any Config field (repeatable), e.g. --set width=64 --set lr=0.1")
    ap.add_argument("--list-presets", action="store_true")
    a = ap.parse_args(argv)

    if a.list_presets:
        for name, p in PRESETS.items():
            print(f"  {name:11} {p.get('arch','mlp'):6} {p.get('dataset','synthetic'):10} "
                  f"steps={p.get('steps','?')}")
        return

    cfg = Config.from_preset(a.preset)
    if a.steps is not None:
        cfg.steps = a.steps
    cfg = _apply_overrides(cfg, a.set)
    if a.run_name:
        cfg.run_name = a.run_name
    if a.wandb:
        cfg.wandb = 1
    if a.wandb_project:
        cfg.wandb_project = a.wandb_project

    import torch
    dtype = {"float32": torch.float32, "float64": torch.float64}.get(a.dtype)
    name = run_name(cfg)
    out = a.out or os.path.join(a.out_root, name)
    os.makedirs(out, exist_ok=True)

    print(f"running preset={a.preset} run={name} on {a.device} → {out}")
    logger = RunLogger(cfg, out, use_wandb=bool(cfg.wandb))
    res = run_job(cfg, device=a.device, dtype=dtype, cifar_dir=a.cifar_dir, progress=True,
                  on_step=logger.log_step)

    torch.save(res, os.path.join(out, "results.pt"))
    summary = {"meta": res["meta"], "config": dataclasses.asdict(cfg),
               "final": {k: (v[-1] if isinstance(v, list) and v and not isinstance(v[0], list) else None)
                         for k, v in res["series"].items()}}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    files = []
    if not a.no_plots:
        from . import plots
        files = plots.save_panels(res, out)
        print("panels:", ", ".join(os.path.basename(f) for f in files))
    logger.finish(res, files)
    m, s = res["meta"], res["series"]
    tail = f"  final loss={s['loss'][-1]:.4e}"
    if "test_loss" in s:
        tail += f"  test={s['test_loss'][-1]:.4e}"
    if "sharpness" in s:
        tail += f"  sharpness={s['sharpness'][-1]:.3f} (2/lr={m['thr']:.3f})"
    if m.get("sections_skipped"):
        tail += f"  skipped={m['sections_skipped']}"
    print(f"done: p={m['p']} wall={m['wall_sec']:.1f}s" + tail)


if __name__ == "__main__":
    main()
