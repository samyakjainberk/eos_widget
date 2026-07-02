"""
eos_lab — an interpretable Python port of the Progressive-Sharpening / Edge-of-Stability widget.

A standalone mirror of `server.py` (the webpage's GPU backend) and `index.html`, organised so each
module maps to a part of the webpage. Run jobs on the GPU yourself — no browser, no HTTP server.

Quick start:
    from eos_lab import Config, run_job
    res = run_job(Config.from_preset("msample"), device="cuda:0", progress=True)
    print(res["series"]["sharpness"][-1], "vs 2/lr =", res["meta"]["thr"])

Module map (each ↔ server.py + index.html):
    rng          mulberry32/gauss            (browser-identical seeding)
    models       architectures + losses + autograd grads/HVPs
    linalg       Lanczos / SLQ / principal angles      (§1–§3, §5, §6 primitives)
    data         synthetic / CIFAR-10 / sorting + init
    config       Config dataclass + PRESETS            (the control panel)
    diagnostics  per-step section computations         (§1,§2,§3,§5,§6)
    train        GD loop + run_job                     (Run button)
    plots        matplotlib panels                     (the plot rows)
    cli          command-line runner
"""
from .config import Config, PRESETS
from .train import run_job, resolve_device
from .models import build_model, build_loss
from . import diagnostics, linalg, data, models, plots, prediction

__all__ = ["Config", "PRESETS", "run_job", "resolve_device",
           "build_model", "build_loss", "diagnostics", "linalg", "data", "models", "plots", "prediction"]
