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
# A SMALL, SYSTEMATIC set of live time-series, grouped into NUMBERED sections that sort in analysis order in
# the wandb sidebar:  1_loss · 2_sharpness · 3_residual · 4_top_eigenvalues · 5_theory_sigma1.  wandb makes
# one chart per metric and groups them by the text before the first "/", so this is ~12 charts in 5 tidy
# sections (down from the old ~24 across 9). Everything else — the full top-/bottom-n spectra, the spectral
# edges, §4/§4b-d projections, §6 rotation, §7/§7a/§8, and all five theory columns of §9/§9b/§9c/§9d — lives
# in the rendered panel images logged once at finish() under "panels/" (the static plots show the full detail).

def _isnum(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


# clean wandb metric name  <-  (record key, vector index or None for a scalar)
_METRICS = [
    ("1_loss/train",                       "loss",       None),
    ("1_loss/test",                        "test_loss",  None),
    ("2_sharpness/lambda_max",             "sharpness",  None),   # the EoS headline: λmax(∇²L) climbing to 2/η
    ("2_sharpness/edge_2_over_eta",        "thr",        None),
    ("3_residual/rms",                     "resid_rms",  None),
    ("4_top_eigenvalues/loss_Hessian",     "lossH_top",  0),      # loss-Hessian = G + S decomposition
    ("4_top_eigenvalues/Gauss_Newton_G",   "G_top",      0),
    ("4_top_eigenvalues/residual_term_S",  "S_top",      0),
]
# §9 theory: the measured σ₁ + three representative predictions (single-sample Eq-13, compact multi Eq-22,
# full-γτz-decomposition multi Eq-29). Eq-21/Eq-23 and the §9b/§9c/§9d variants are in the panel images.
_THEORY_COLS = [(0, "Eq13_single"), (2, "Eq22_multi"), (4, "Eq29_multi")]


def flatten_rec(rec):
    """Per-step record → a small, grouped {metric: float} dict for wandb.log (numbered sections)."""
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
    # §9 theory vs empirical sharpness σ₁ — measured + the representative predicted equations, one section
    thP, thA = rec.get("thP"), rec.get("thA")
    if thA:
        for x in thA:                       # thA[1..4] (multi) or thA[0] (single) all hold the measured σ₁
            if _isnum(x):
                out["5_theory_sigma1/measured"] = float(x)
                break
    if thP:
        for i, nm in _THEORY_COLS:
            if i < len(thP) and _isnum(thP[i]):
                out["5_theory_sigma1/predicted_" + nm] = float(thP[i])
    # §10 cubic approximation (only when s17 on): measured NTK σ₁ + cubic (Eq-47 single / Eq-51 multi, +PSD)
    # and the quadratic (Eq-51 without the η² term) prediction. The full ±PSD/drift detail is in the panel images.
    for nm, key in (("measured_NTK", "cActN"), ("cubic_Eq47_single", "c47p"),
                    ("cubic_Eq51_multi", "c51p"), ("quadratic_Eq51_no_eta2", "c51np")):
        v = rec.get(key)
        if _isnum(v):
            out["6_cubic_sigma1/" + nm] = float(v)
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
