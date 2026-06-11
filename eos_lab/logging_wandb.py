"""
Run naming + Weights & Biases logging.

`run_name(cfg)` builds a deterministic, human-readable name from the hyperparameters — used both for
the wandb run name and the local output folder, so a run is self-identifying. `RunLogger` streams the
per-step diagnostics to wandb (one point per eig-tick) and, at the end, logs every rendered panel image.

Everything degrades gracefully: with `wandb=0` (or wandb not installed / no API key) the local saving
and plotting still happen; only the wandb calls are skipped.
"""
import math
import os


def _fmt(v):
    """Compact, filesystem-safe number formatting: 0.001 → '0p001', 1e-4 → '1em4'."""
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            s = f"{v:.0e}".replace("e-0", "em").replace("e-", "em").replace("e+0", "e").replace("e+", "e")
        else:
            s = ("%g" % v)
        return s.replace(".", "p").replace("-", "m")
    return str(v)


def run_name(cfg):
    """Deterministic run id from the hyperparameters (dataset, arch, size, lr, samples, seed, …)."""
    if getattr(cfg, "run_name", ""):
        return cfg.run_name
    parts = [cfg.dataset, cfg.arch]
    if cfg.arch in ("mlp",):
        parts += [cfg.act, f"d{cfg.depth}", f"w{cfg.width}"]
    if cfg.arch in ("cnn", "vgg11"):
        parts += [f"chmul{_fmt(cfg.chmul)}"]
    if cfg.arch == "gpt":
        parts += [f"dm{cfg.dmodel}", f"nl{cfg.nlayer}", f"seq{cfg.seqlen}"]
    parts += [cfg.loss, f"n{cfg.nsamp}", f"lr{_fmt(cfg.lr)}", f"init{_fmt(cfg.init)}",
              f"st{cfg.steps}", f"seed{cfg.seed}"]
    return "_".join(str(p) for p in parts)


# ----------------------------------------------------------------------------- per-step → wandb metrics
# A small, curated, GROUPED set of human-readable time-series — NOT the ~120 cryptic per-index series the
# raw diagnostics expand to (H_top/0, jt/0, q9t/0, ntkR/0, …). wandb groups charts by the text before the
# first "/", so the names below land in a handful of tidy sections (loss · sharpness · top_eigenvalue · …).
# The full top-/bottom-n spectra, NTK alignments, FH-SVD and §4b–§4d projections are deliberately NOT
# streamed here — that detail lives in the rendered panel images logged once at finish().

def _isnum(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


# clean wandb metric name  <-  (record key, vector index or None for a scalar)
_METRICS = [
    ("loss/train",                         "loss",       None),
    ("loss/test",                          "test_loss",  None),
    ("sharpness/sharpness",                "sharpness",  None),
    ("sharpness/edge_2_over_lr",           "thr",        None),
    ("residual/mean",                      "resid_mean", None),
    ("residual/rms",                       "resid_rms",  None),
    ("top_eigenvalue/loss_Hessian",        "lossH_top",  0),
    ("top_eigenvalue/function_Hessian",    "H_top",      0),
    ("top_eigenvalue/Gauss_Newton",        "G_top",      0),
    ("top_eigenvalue/residual_term",       "S_top",      0),
    ("bottom_eigenvalue/loss_Hessian",     "lossH_bot",  0),
    ("bottom_eigenvalue/function_Hessian", "H_bot",      0),
    ("H_spectral_edge/max",                "H_edge_max", None),
    ("H_spectral_edge/min",                "H_edge_min", None),
    ("J_projection/onto_top_H_eigvec",     "jtN",        0),
    ("J_projection/onto_bottom_H_eigvec",  "jbN",        0),
]
_THEORY_EQS = ["Eq13_single_sample", "Eq21_multi_sample", "Eq22_multi_sample",
               "Eq23_multi_sample", "Eq29_multi_sample"]   # thP cols 0-4 (Eq-27 was replaced by Eq-22; Eq-23 added)


def flatten_rec(rec):
    """Per-step record → a small dict of grouped, readable {metric: float} for wandb.log."""
    out = {}
    for name, key, idx in _METRICS:
        v = rec.get(key)
        if v is None:
            continue
        if idx is not None:
            if not isinstance(v, (list, tuple)) or len(v) <= idx:
                continue
            v = v[idx]
        if _isnum(v):
            out[name] = float(v)
    # §6 eigenspace rotation — max principal angle (degrees) of the ± subspaces
    for key, nm in (("rot_pos", "positive"), ("rot_neg", "negative")):
        d = rec.get(key)
        if d and _isnum(d.get("mx")):
            out["eigenspace_rotation/" + nm + "_deg"] = float(d["mx"])
    # §9 theory vs empirical sharpness σ₁ — predicted per equation + the single measured actual
    thP, thA = rec.get("thP"), rec.get("thA")
    if thP:
        for i, eq in enumerate(_THEORY_EQS):
            if i < len(thP) and _isnum(thP[i]):
                out["theory_sigma1/predicted_" + eq] = float(thP[i])
    if thA:
        for x in thA:
            if _isnum(x):
                out["theory_sigma1/actual"] = float(x)
                break
    return out


class RunLogger:
    """Optional wandb run wrapper. Use as: logger = RunLogger(cfg, outdir); run_job(on_step=logger.log_step);
    logger.finish(results, panel_files)."""
    def __init__(self, cfg, outdir, use_wandb=False, meta=None):
        self.cfg, self.outdir, self.use_wandb = cfg, outdir, bool(use_wandb)
        self.run = None
        if self.use_wandb:
            try:
                import wandb
                import dataclasses
                conf = dataclasses.asdict(cfg)
                if meta:
                    conf = {**conf, **{f"meta_{k}": v for k, v in meta.items()}}
                self.run = wandb.init(
                    project=cfg.wandb_project or "eos_lab",
                    entity=(cfg.wandb_entity or None),
                    name=run_name(cfg), config=conf,
                    dir=outdir, reinit=True)
            except Exception as e:                       # never let logging kill a run
                print(f"  [wandb] disabled ({type(e).__name__}: {e})")
                self.use_wandb = False

    def log_step(self, rec, meta):
        if not self.use_wandb or self.run is None:
            return
        try:
            import wandb
            wandb.log(flatten_rec(rec), step=int(rec["t"]))
        except Exception:
            pass

    def finish(self, results, panel_files):
        if not self.use_wandb or self.run is None:
            return
        try:
            import wandb
            wandb.log({"panels/" + os.path.splitext(os.path.basename(f))[0]: wandb.Image(f)
                       for f in panel_files})
            m = results["meta"]
            wandb.summary.update({"final_loss": results["series"].get("loss", [None])[-1],
                                  "wall_sec": m.get("wall_sec"), "p": m.get("p"),
                                  "sections_skipped": m.get("sections_skipped")})
            wandb.finish()
        except Exception:
            pass
