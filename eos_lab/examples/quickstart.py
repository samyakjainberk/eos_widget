"""
quickstart — three ways to use eos_lab. Run:  python eos_lab/examples/quickstart.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eos_lab import Config, run_job, PRESETS, plots

# 1) a preset, as-is
print("presets:", list(PRESETS))
res = run_job(Config.from_preset("msample", steps=300), device="auto", progress=True)
print("final sharpness:", res["series"]["sharpness"][-1], "vs 2/lr =", res["meta"]["thr"])

# 2) a fully custom config (every webpage control is a field)
cfg = Config(dataset="synthetic", arch="mlp", act="tanh", depth=4, width=40,
             nsamp=20, fixedx=0, lr=0.4, steps=400, seed=3, s5=0, s6=1)
res2 = run_job(cfg, device="auto")
print("custom run: p =", res2["meta"]["p"], " final loss =", res2["series"]["loss"][-1])

# 3) reach into a single primitive (matrix-free top eigenvalue of the loss Hessian)
import torch
from eos_lab.models import build_model, build_loss, hvp_L
from eos_lab.linalg import lanczos_extreme_vals
dev = res2["meta"]["device"]
model = build_model("mlp", 6, 1, Config(width=10, depth=3).P(), torch.device(dev), torch.float32)
loss = build_loss("mse")
th = model.init_theta(1, 0.1)
X = torch.randn(8, 6, device=dev); Y = torch.randn(8, 1, device=dev)
top, bot = lanczos_extreme_vals(lambda v: hvp_L(model, loss, th, X, Y, v),
                                model.p, 3, 28, 0x5EED1, torch.device(dev), torch.float32)
print("top-3 loss-Hessian eigenvalues at init:", top)

# render panels for run #1
files = plots.save_panels(res, "runs/quickstart")
print("wrote:", files)
