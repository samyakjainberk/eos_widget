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

from .models import build_model, build_loss, grad_loss, opt_dir, MlpModel
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
    if cfg.dataset == "cifar2":
        return 3072, 1                         # 2-class CIFAR cast as SCALAR regression (±1)
    if cfg.dataset == "mnist":
        return 3072, 10                        # MNIST padded to 3×32×32 (drop-in for cifar-shaped MLP/CNN/VGG)
    if cfg.dataset == "mnist2":
        return 3072, 1                         # 2-class MNIST cast as SCALAR regression (±1)
    if cfg.dataset == "sorting":
        return cfg.seqlen, cfg.seqlen
    if cfg.dataset == "owt":
        return cfg.seqlen, cfg.vocab       # in_dim = block size (tokens/seq), out_dim = vocab
    if cfg.dataset in ("chebyshev", "chebyshev2"):
        return 1, 1                        # scalar x → scalar T_k(x)
    if cfg.dataset == "ksparse":
        return cfg.indim, 1                # n ±1 bits in → scalar ±1 parity out (gpt: read as a length-n bit sequence, mean-pooled)
    if cfg.dataset == "anglepair":
        return max(2, cfg.indim), 1        # two iid-Gaussian samples (norm/angle controlled) → scalar ±1
    if cfg.dataset == "agf":
        return cfg.indim, 1                # alternating gradient flows: scalar linear regression y=β*·x
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
    th0 = th.detach().clone()      # §16 standalone optimizer starts from θ₀ (the GD loop reassigns `th`)
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

    # ── §22 (s30) / §23 (s31): quadratic-Taylor surrogate REPLACE drivers ──────────────────────────────
    # MIRRORS server.run_stream's `do_quad_sur` branch (server.py:3086-3120): when on, these REPLACE the real
    # GD loop entirely — GD is run on the frozen-Q quadratic surrogate from a freeze point θ_f (§22: θ_t after
    # s30t real steps; §23: θ₀ with a random rank-s31r Q), and §1/§2/§3 + §20 + (if s23/§15-on) the full §15
    # are computed ALONG the surrogate. Gate: (s30 or s31) and MSE and dataset≠owt and M=N·d_out ≤ grid3dcap.
    t0 = time.time()
    M_surr = N * out_dim
    grid3dcap = max(1, getattr(cfg, "grid3dcap", 500))
    do_quad_sur = ((getattr(cfg, "s30", 0) or getattr(cfg, "s31", 0)) and loss.name == "mse"
                   and cfg.dataset != "owt" and M_surr <= grid3dcap)
    if do_quad_sur:
        from .sec22_23 import quad_surrogate_driver
        ee = max(1, cfg.eigevery)
        sec20k = max(1, getattr(cfg, "s28k", 40))   # §20/§22/§23 share the M_r top-K⊕bottom-K count
        if progress:
            kind = "frozen-Q (§22)" if getattr(cfg, "s30", 0) else "random-Q (§23)"
            print(f"  §22/§23 surrogate REPLACE mode: {kind} — {cfg.steps} surrogate steps from the freeze point")
        sec2223 = list(quad_surrogate_driver(
            model, loss, th0, X, Y, N, out_dim, cfg.lr, cfg.steps, ee, sec20k,
            diag.n, diag.mV, diag.nResid, P.get("s1", 0), P.get("s2", 0), P.get("s3", 0), P.get("gs", True),
            opt=cfg.optimizer, s15=getattr(cfg, "s23", 0), divreg=float(getattr(cfg, "divreg", 0.3)),
            s30=getattr(cfg, "s30", 0), s30t=getattr(cfg, "s30t", 0),
            s31=getattr(cfg, "s31", 0), s31r=getattr(cfg, "s31r", 200),
            s31scale=getattr(cfg, "s31scale", 1.0), seed=cfg.seed))
        meta["wall_sec"] = time.time() - t0
        meta["surrogate"] = "g22" if getattr(cfg, "s30", 0) else "g23"
        return {"meta": meta, "config": cfg, "history": [], "series": {},
                "sec16": None, "sec17": None, "sec2223": sec2223}

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
            th = th - cfg.lr * opt_dir(model, grad_loss(model, loss, th, Xb, Yb)[0], cfg.optimizer)

    # ── §16: standalone curvature-aligned per-residual-sign optimizer (own iteration axis), run from θ₀ ──
    # MULTI-OUTPUT: the effective samples are the M = N·d_out outputs (residual / Jacobian-row dim), so cifar10 (d=10) /
    # sorting-GPT (d=seqlen) work too. Gate on M=N·d (jac_cols = M backward) + M·p memory; any arch, MSE only.
    sec16 = None
    M16 = N * out_dim
    if getattr(cfg, "s24", 0) and loss.name == "mse" and M16 <= 256 and M16 * model.p <= 300_000_000 and N <= 64:
        from .sec16 import sec16_driver, sec16_chebyshev_testset, sec16_chebyshev2_testset
        if progress:
            print(f"  §16: standalone optimizer — {cfg.s24warm} GD warmup + {cfg.s24iter} iters from θ₀  (M=N·d={M16})")
        if cfg.dataset == "chebyshev":                                  # §16 Panel 5 held-out test set (midpoints of the grid)
            Xt16, Yt16 = sec16_chebyshev_testset(N, cfg.degree, dev, dtype)
        elif cfg.dataset == "chebyshev2":
            Xt16, Yt16 = sec16_chebyshev2_testset(N, cfg.degree, dev, dtype)
        else:                                                           # all others: real held-out split (cifar/mnist/sorting/synthetic/const) or None
            _ts16 = make_test_set(model, P, cfg.dataset, N, in_dim, out_dim, dev, dtype, cifar_dir)
            Xt16, Yt16 = _ts16 if _ts16 is not None else (None, None)
        _k16 = max(1, getattr(cfg, 's24k', 32))
        sec16 = list(sec16_driver(model, loss, th0, X, Y, cfg.lr, cfg.s24warm, cfg.s24iter,
                                  min(max(1, cfg.neig), max(1, model.p // 2)),
                                  min(model.p, max(4 * _k16 + 32, 2 * cfg.neig + 16)), cfg.seed, 1e-12,
                                  getattr(cfg, 's24grid', 0.1), getattr(cfg, 's24ares', 0.01),
                                  Xt16, Yt16, getattr(cfg, 's24base', 0), _k16))   # baselines A/B/C/D/E if s24base

    # §17 — PER-SAMPLE function-Hessian variant of §16 (each sample uses its OWN Q_k eigvec). O(M²) per-sample
    # Lanczos ⇒ gated to a SMALLER M than §16. Reuses §16's hyperparameters + held-out test sets (same observables).
    sec17 = None
    if getattr(cfg, "s25", 0) and loss.name == "mse" and M16 <= 256 and M16 * model.p <= 300_000_000 and N <= 64:
        from .sec17 import sec17_driver
        from .sec16 import sec16_chebyshev_testset, sec16_chebyshev2_testset
        if progress:
            print(f"  §17: per-sample optimizer — {cfg.s24warm} GD warmup + {cfg.s24iter} iters from θ₀  (M=N·d={M16})")
        if cfg.dataset == "chebyshev":
            Xt17, Yt17 = sec16_chebyshev_testset(N, cfg.degree, dev, dtype)
        elif cfg.dataset == "chebyshev2":
            Xt17, Yt17 = sec16_chebyshev2_testset(N, cfg.degree, dev, dtype)
        else:
            _ts17 = make_test_set(model, P, cfg.dataset, N, in_dim, out_dim, dev, dtype, cifar_dir)
            Xt17, Yt17 = _ts17 if _ts17 is not None else (None, None)
        _k17 = max(1, getattr(cfg, 's24k', 32))
        sec17 = list(sec17_driver(model, loss, th0, X, Y, cfg.lr, cfg.s24warm, cfg.s24iter,
                                  min(max(1, cfg.neig), max(1, model.p // 2)),
                                  min(model.p, max(4 * _k17 + 32, 2 * cfg.neig + 16)), cfg.seed, 1e-12,
                                  getattr(cfg, 's24grid', 0.1), getattr(cfg, 's24ares', 0.01),
                                  Xt17, Yt17, getattr(cfg, 's25base', 0), _k17))   # baselines A/B/C/D/E if s25base

    meta["wall_sec"] = time.time() - t0
    return {"meta": meta, "config": cfg, "history": history, "series": _to_series(history),
            "sec16": sec16, "sec17": sec17, "sec2223": None}


# every per-step key that is a list-of-numbers and should become one track-per-index in `series`
_VECTOR_KEYS = [
    "resid_head",                                                    # §1 per-sample residual y−f (first nResid)
    "H_top", "H_bot", "lossH_top", "lossH_bot", "G_top", "G_bot", "S_top", "S_bot",
    "jt", "jb", "jtN", "jbN", "jrt", "jrb",                           # §4 (incl. phase-3 residual-dominance |⟨J·r,u⟩|·σ)
    "ntkR", "ntkH", "ntkGs", "ntkGsA", "ntkGl", "ntkGlA",            # §7a (+ Δλ·Δr products & running avgs)
    "fhEvT", "fhEvB", "jhe1", "jhe2", "jhe1b", "jhe2b",              # §7
    "jh2e1", "jh2e2", "jh2e1b", "jh2e2b",
    "g2J", "g2Jn",                                                    # §8
    "q9t", "q9b", "q9tN", "q9bN", "q9tR", "q9bR", "q9gt", "q9gb",     # §4b
    "q10", "q10N",                                                    # §4c
    "d4Ht", "d4Hb", "d4Qt", "d4Qb",                                   # §4d
    "thP", "thA", "thPpsd",                                           # §9 / §9b (length-5, may contain None)
    "thP_d", "thPpsd_d",                                              # §9d / §9d-c (quad-self-residual predictions)
]


def _to_series(history):
    """Pivot the per-step records into plottable column arrays (only the keys that appear)."""
    if not history:
        return {}
    steps = [r["t"] for r in history]
    s = {"t": steps}
    scalar_keys = ["loss", "test_loss", "sharpness", "thr", "resid_mean", "resid_rms",
                   "H_edge_max", "H_edge_min", "dim_pos", "dim_neg", "thAH",
                   "c47", "c47p", "c51", "c51p", "c51n", "c51np",   # §10 cubic predictions
                   "cActN", "cActH", "cdQ", "cdJ"]                  # §10 cubic actuals + ‖ΔQ‖/‖ΔJ‖ drift
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
