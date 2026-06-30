# Batch runs on SLURM / GPU, analyse in the browser later

Run the widget's full computation **headless** (no live server, no tab open), save **every section's
records** to a file, then **load it back into the widget** to scrub all the plots offline. Fan many
runs out across SLURM and your GPUs in parallel and analyse them whenever.

Three pieces:

| | what |
|---|---|
| `capture_run.py` | headless runner — same compute as the GPU backend's `/run`, writes a `.json` capture |
| `run_capture.sh` | SLURM batch wrapper around `capture_run.py` (one job → one capture file) |
| widget **⬆ load run** | loads a `.json` capture and replays it into every plot (also restores the controls) |

By default the section toggles **s1–s27 plus §20/§21 (s28/s29) are ON** (so a capture records that whole
toggle list, covering **§1–§21**), while the **§16/§17 baselines (s24base/s25base) and §22–§25 (s30–s33)
default OFF** — opt into them per-run with `--set sNN=1`. Whether a section actually produces data still
depends on its gates (multi-sample / small-N / loss type) — exactly like the live widget, which shows a
"not produced" note for any gated-out section.

## 1. Capture (headless)

Locally (GPU auto-detected; CPU works but the heavy sections are slow — use a GPU):

```bash
# everything on, defaults (small multi-sample MLP so every section produces):
python capture_run.py --out runs_captured/demo.json

# a real config — flags are plain widget params:
python capture_run.py --dataset cifar10 --arch vgg11 --nsamp 25 --steps 400 \
                      --device cuda:0 --out runs_captured/cifar_vgg.json

# turn a couple of heavy sections off, keep the rest (any param via --set key=value):
python capture_run.py --set s17=0 --set s19=0 --out runs_captured/light.json
```

On SLURM (one job per capture):

```bash
sbatch --export=ALL,ARGS="--dataset cifar10 --arch vgg11 --nsamp 25 --steps 400 \
                          --out runs_captured/cifar_vgg.json" run_capture.sh
```

A sweep — submit many jobs, each its own file (they run in parallel on whatever GPUs are free):

```bash
for lr in 0.02 0.05 0.1; do
  sbatch --export=ALL,ARGS="--dataset cifar10 --arch mlp --nsamp 25 --steps 400 \
         --lr $lr --out runs_captured/cifar_mlp_lr${lr}.json" run_capture.sh
done
# or use the helper:  ./sweep_capture.sh
```

## 2. Load in the widget

Open the widget, then **Compute → "⬆ load run"** and pick the `.json`. It restores the run's controls
and replays every record into the plots — §1–§9 time-series, §11–§15 cubes/grids (with the snapshot
sliders), and the §16/§17 optimizer panels (main + 5 baselines). No server needed; this works on the
static page too.

You can also **⬇ save run** to download the most recent **GPU-backend** run from the browser (the same
capture format), then reload it later.

### Loading is fast and never hangs

An all-sections capture is large because §14 alone is ~80% of it (45 N³ cubes per snapshot), even though
§14 only ever shows time-series. So:

- Every finished capture also gets a **`<name>.min.json`** — the §14 cubes reduced to the exact
  time-series stats the browser plots (≈5× smaller, e.g. 200 MB → 40 MB). `capture_run.py` writes it
  automatically; **⬆ load from server** loads the `.min` sibling automatically (you don't pick it). The
  plots are bit-identical — the raw cubes were never displayed. To compact an old capture by hand:
  `python compact_capture.py runs_captured/<name>.json` (non-destructive → `<name>.min.json`).
- Loading **streams** the file and renders on a frame budget — records are parsed in ≤64 KB slices and
  plots render a few per frame — so the tab stays responsive for **any** size (no multi-second freeze,
  no big `JSON.parse` block). The full uncompacted `.json` still loads too (just a bit slower).

### Notes
- **Late-section params** (set via `--set`): `s28k` (§20 — # eigenpairs per side of `M_r`, default 40),
  `s29k` (§21 — NTK top-K & `M_r` top-K⊕bottom-K, default 40), and the §22/§23 surrogate controls
  `s30t` (§22 — iteration `t` to freeze `Q`, default 0), `s31r` (§23 — random-Hessian rank, default 200),
  `s31scale` (§23 — random-Hessian spectral-norm scale, default 1.0).
- **§22/§23 surrogate captures REPLACE the core plots.** Turning on `s30` (§22 frozen-Q) or `s31` (§23
  random-Q) makes the whole run a quadratic-Taylor surrogate: the capture's §1 (loss/sharpness/residual),
  §2/§3 spectra and §20 `M_r` show the **surrogate trajectory, not the real model** (§4–§21 / §24 are off,
  and the loaded capture shows a warning banner). §22 and §23 are mutually exclusive (frozen-Q wins).
- Output is plain `.json` (loads everywhere). `--gzip` writes `.json.gz` (smaller, but the browser
  loader needs `pako` bundled — keep plain `.json` unless you've added it).
- A capture is self-contained: `{format, params, record_counts, records}`. `params` is the query-style
  control dict used to restore the UI; `records` is the exact backend stream.
- Browser-compute (the "browser (this tab)" mode) is **not** record-based, so **⬇ save run** only
  captures **GPU-backend** runs. Use `capture_run.py` / `run_capture.sh` for headless captures.
