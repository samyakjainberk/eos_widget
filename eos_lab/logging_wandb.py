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


# ----------------------------------------------------------------------------- per-step flattening
_SCALARS = ["loss", "test_loss", "sharpness", "thr", "resid_mean", "resid_rms",
            "H_edge_max", "H_edge_min", "dim_pos", "dim_neg"]
_VECTORS = ["H_top", "H_bot", "lossH_top", "lossH_bot", "G_top", "G_bot", "S_top", "S_bot",
            "jt", "jb", "jtN", "jbN", "ntkR", "ntkH", "fhEvT", "fhEvB",
            "jhe1", "jhe2", "jhe1b", "jhe2b", "jh2e1", "jh2e2", "jh2e1b", "jh2e2b",
            "g2J", "g2Jn", "q9t", "q9b", "q9tN", "q9bN", "q9tR", "q9bR", "q9gt", "q9gb",
            "q10", "q10N", "d4Ht", "d4Hb", "d4Qt", "d4Qb", "thP", "thA", "thPpsd"]


def flatten_rec(rec):
    """Per-step record → flat {metric: float} dict for wandb.log (vectors expand to base/i)."""
    out = {}
    for k in _SCALARS:
        if k in rec and rec[k] is not None and _isnum(rec[k]):
            out[k] = float(rec[k])
    for base in _VECTORS:
        v = rec.get(base)
        if not v:
            continue
        for i, x in enumerate(v):
            if x is not None and _isnum(x) and not (isinstance(x, float) and math.isnan(x)):
                out[f"{base}/{i}"] = float(x)
    for base in ("rot_pos", "rot_neg"):
        d = rec.get(base)
        if d:
            out[base + "_max"] = float(d["mx"])
            out[base + "_mean"] = float(d["mn"])
    return out


def _isnum(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


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
