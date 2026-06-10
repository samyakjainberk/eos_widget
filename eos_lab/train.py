"""
Training driver — full-batch gradient descent + streaming diagnostics.

MIRRORS:  server.run_stream  (the GD loop and the meta/step record structure)  ↔  index.html (Run).

`run_job(cfg)` builds the model/loss/data, runs GD for `steps`, computes the §1/2/3/5/6 diagnostics
every `eigevery` steps, and returns a results dict (meta + per-step history + tidy 'series' arrays).
This is the single entry point the CLI, SLURM script, and notebooks all call.
"""
import copy
import math
import time

import numpy as np
import torch

from .models import build_model, build_loss, grad_loss
from .data import init_data_theta, make_test_set
from .diagnostics import Diagnostics

PMAX = 10_000_000


def resolve_device(device):
    """device: None|'auto'|'cpu'|'cuda'|'cuda:N' → torch.device. 'auto' = freest CUDA GPU."""
    if isinstance(device, torch.device):
        return device
    if device in (None, "auto"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        best, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            try:
                free, _ = torch.cuda.mem_get_info(i)
            except Exception:
                free = 0
            if free > best_free:
                best_free, best = free, i
        return torch.device(f"cuda:{best}")
    return torch.device(device)


def effective_dims(cfg):
    """Input/output dims actually fed to the model (CIFAR=3072/10, sorting=seqlen, owt=block/vocab)."""
    if cfg.dataset == "cifar10":
        return 3072, 10
    if cfg.dataset == "sorting":
        return cfg.seqlen, cfg.seqlen
    if cfg.dataset == "owt":
        return cfg.seqlen, cfg.vocab       # in_dim = block size (tokens/seq), out_dim = vocab
    if cfg.dataset == "chebyshev":
        return 1, 1                        # scalar x → scalar T_k(x)
    return cfg.indim, cfg.outdim


def run_job(cfg, device=None, dtype=None, cifar_dir=None, progress=False, on_step=None):
    """Run one EoS/PS job. Returns {'meta':..., 'history':[rec,...], 'series':{...}, 'config':...}.
    on_step(rec, meta) is called after each diagnostics tick (used for live wandb logging)."""
    dev = resolve_device(device)
    if dtype is None:
        dtype = torch.float32 if dev.type == "cuda" else torch.float64
    torch.set_grad_enabled(False)

    in_dim, out_dim = effective_dims(cfg)
    P = cfg.P()
    model = build_model(cfg.arch, in_dim, out_dim, P, dev, dtype)
    loss = build_loss(cfg.loss)
    if model.p > PMAX:
        raise ValueError(f"p={model.p} exceeds cap {PMAX}; reduce width/depth/chmul.")

    N = cfg.nsamp
    th, X, Y, pos_rows, neg_rows = init_data_theta(model, P, cfg.dataset, N, in_dim, out_dim,
                                                   dev, dtype, cifar_dir)
    n_test = cfg.n_test if cfg.n_test > 0 else min(max(N, 256), 1000)
    test = make_test_set(model, P, cfg.dataset, n_test, in_dim, out_dim, dev, dtype, cifar_dir)

    # minibatch size: each GD step + that step's diagnostics run on a `bs`-sample sample of the pool.
    # bs==N ⇒ deterministic full-batch GD (the default for every dataset except owt) — identical to
    # before. The diagnostics' own N (loss normalisation, M, Lanczos sizes) must equal bs, so build it
    # from a cfg copy whose nsamp is bs.
    bs = N if cfg.batch <= 0 else min(cfg.batch, N)
    full_batch = bs >= N
    diag_cfg = copy.copy(cfg)
    diag_cfg.nsamp = bs
    diag = Diagnostics(model, loss, diag_cfg, dev, dtype)

    meta = {"p": model.p, "arch": cfg.arch, "dataset": cfg.dataset, "loss": loss.name,
            "thr": 2.0 / cfg.lr, "device": str(dev), "dtype": str(dtype).replace("torch.", ""),
            "nsamp": N, "batch": bs, "steps": cfg.steps, "lr": cfg.lr, "M": diag.M,
            "multi_ok": diag.multi_ok, "n_test": (test[0].shape[0] if test is not None else 0),
            "sections_skipped": diag.sections_skipped(),
            "hess_per_sample": True}   # §1/§2/§3 H eigenvalues are divided by N (per-sample scale)
    if test is not None and progress and diag.sections_skipped():
        print(f"  note: skipping {diag.sections_skipped()} (M={diag.M}, p={model.p} too large)")
    if not full_batch and progress:
        print(f"  minibatch SGD: batch={bs} of nsamp={N} (fresh sample every step, used for GD + diagnostics)")

    def minibatch(t):
        """A fresh size-bs sample of the pool for step t (used for BOTH the GD step and diagnostics).
        Full-batch ⇒ the whole pool every step (deterministic). Returns (Xb, Yb, pos, neg)."""
        if full_batch:
            return X, Y, pos_rows, neg_rows
        r = np.random.RandomState((cfg.seed * 1000003 + t) & 0x7FFFFFFF)
        idx = torch.as_tensor(r.choice(N, size=bs, replace=False), dtype=torch.long, device=dev)
        return X[idx], Y[idx], None, None      # §4d sign-groups undefined under sub-sampling

    history = []
    ee = max(1, cfg.eigevery)
    t0 = time.time()
    for t in range(cfg.steps + 1):
        Xb, Yb, pr, nr = minibatch(t)
        if t % ee == 0:
            rec = diag.compute(th, Xb, Yb, t, pr, nr)
            if test is not None:                     # held-out test loss (same loss fn)
                with torch.no_grad():
                    rec["test_loss"] = float(loss.value(model.forward(th, test[0]), test[1], test[0].shape[0]))
            history.append(rec)
            if not math.isfinite(rec["loss"]) or rec["loss"] > 1e10:
                rec["diverged"] = True
                if progress:
                    print(f"  diverged at step {t} (loss={rec['loss']:.2e})")
                break
            if progress and (t % (ee * 10) == 0):
                sps = (t + 1) / max(time.time() - t0, 1e-9)
                msg = f"  step {t}/{cfg.steps}  loss={rec['loss']:.4e}"
                if "test_loss" in rec:
                    msg += f"  test={rec['test_loss']:.4e}"
                if "sharpness" in rec:
                    msg += f"  sharpness={rec['sharpness']:.3f}  (2/lr={meta['thr']:.3f})"
                print(msg + f"  [{sps:.1f} steps/s]")
            if on_step is not None:
                on_step(rec, meta)
        if t < cfg.steps:
            th = th - cfg.lr * grad_loss(model, loss, th, Xb, Yb)[0]
            if t % 25 == 0 and th.is_cuda:     # release unused cached GPU blocks (mirror server.run_stream)
                torch.cuda.empty_cache()

    meta["wall_sec"] = time.time() - t0
    return {"meta": meta, "config": cfg, "history": history, "series": _to_series(history)}


# every per-step key that is a list-of-numbers and should become one track-per-index in `series`
_VECTOR_KEYS = [
    "H_top", "H_bot", "lossH_top", "lossH_bot", "G_top", "G_bot", "S_top", "S_bot",
    "jt", "jb", "jtN", "jbN",                                         # §4
    "ntkR", "ntkH", "fhEvT", "fhEvB", "jhe1", "jhe2", "jhe1b", "jhe2b",   # §7
    "jh2e1", "jh2e2", "jh2e1b", "jh2e2b",
    "g2J", "g2Jn",                                                    # §8
    "q9t", "q9b", "q9tN", "q9bN", "q9tR", "q9bR", "q9gt", "q9gb",     # §4b
    "q10", "q10N",                                                    # §4c
    "d4Ht", "d4Hb", "d4Qt", "d4Qb",                                   # §4d
    "thP", "thA",                                                     # §9 (length-4, may contain None)
]


def _to_series(history):
    """Pivot the per-step records into plottable column arrays (only the keys that appear)."""
    if not history:
        return {}
    steps = [r["t"] for r in history]
    s = {"t": steps}
    scalar_keys = ["loss", "test_loss", "sharpness", "thr", "resid_mean", "resid_rms",
                   "H_edge_max", "H_edge_min", "dim_pos", "dim_neg", "thAH"]
    for k in scalar_keys:
        if any(k in r for r in history):
            s[k] = [r.get(k, math.nan) for r in history]
    # vector tracks: base[0..n-1] → one column array each (None / missing → NaN)
    for base in _VECTOR_KEYS:
        rows = [r[base] for r in history if r.get(base) is not None]
        if not rows:
            continue
        n = max(len(x) for x in rows)
        s[base] = [[(r[base][j] if (r.get(base) is not None and j < len(r[base])
                                    and r[base][j] is not None) else math.nan)
                    for r in history] for j in range(n)]
    # §6 rotation: max/mean principal angle of the ± subspaces
    for base in ["rot_pos", "rot_neg"]:
        if any(base in r and r[base] for r in history):
            s[base + "_max"] = [(r[base]["mx"] if r.get(base) else math.nan) for r in history]
            s[base + "_mean"] = [(r[base]["mn"] if r.get(base) else math.nan) for r in history]
    return s
