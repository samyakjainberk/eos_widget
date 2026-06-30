# Progressive Sharpening & Edge of Stability

> An interactive companion to ***Understanding Progressive Sharpening*** — train a small net with
> full-batch GD and **watch the curvature climb to the edge of stability live**, with every panel tied
> to the paper's equations.

**The one-liner.** As a net trains, its loss surface keeps getting *sharper* (the top Hessian eigenvalue
climbs). GD is only stable while that sharpness stays below `2/η`, so training drives itself up to that
line and then hovers — the climb is **progressive sharpening**, the hovering is the **edge of stability**.
This tool runs that experiment in front of you and overlays the paper's predictions on what the net
actually does.

👉 **Just want to watch it? Press Run.** The math lives in the in-app captions; this README is the map.

---

## ⚡ Quick start

```bash
# 1) Browser widget — zero backend, runs entirely in JS (also deploys to GitHub Pages as-is)
python3 -m http.server 8000          # → http://localhost:8000/

# 2) GPU backend — same UI, PyTorch compute over SSE; adds real datasets & big nets
python3 server.py --device cuda:0 --port 8756 --cifar-dir data/cifar-10-batches-py
#   then in the page:  Compute → "GPU backend (server.py)"
#   --dtype auto = fp32 on GPU / fp64 on CPU · --devices all for a multi-GPU pool

# 3) Headless Python reference (eos_lab)
python -m eos_lab.cli --list-presets
python -m eos_lab.cli --preset cifar_mlp --device cuda:0     # → results + section*.png
```

`run_serve_localhost.sh` launches the GPU backend on a SLURM A100 and opens a reverse SSH tunnel so the
widget is reachable at `http://localhost:8756/` from your laptop.

---

## 🧩 Backends at a glance

Three interchangeable front-ends share **one matrix-free numerical core** (Hessian-vector products +
Lanczos — no `p×p` matrix is ever formed, so it scales to multi-million-parameter nets).

| Backend | What it is | Precision | Sections | File |
|---|---|---|---|---|
| **Browser widget** | zero-backend; synthetic MLP runs in JS | fp64 | §1–§25 | `index.html` |
| **GPU backend** | same UI, PyTorch over SSE; real datasets + big nets | fp32 GPU / fp64 CPU | §1–§25 | `server.py` |
| **`eos_lab/`** | headless Python package for scripted / SLURM runs | fp64 (fp32 GPU) | **§1–§17 only** | `eos_lab/` |

**Byte-parity invariant.** Data + init use the same `mulberry32` RNG across all three, so at a given seed
the **GD trajectory is bit-identical** everywhere. `server.py`/`index.html` use finite-difference HVPs to
stay byte-parity; `eos_lab` uses exact autograd HVPs (curvature *probes* differ by ~4%, the trajectory
does not). §18–§25 are server/browser only.

---

## 🗺️ Repo map

```
index.html        browser widget (also served by server.py)
server.py         GPU backend — HTTP + SSE, PyTorch
eos_lab/          headless package: config (presets) · models · data · diagnostics · sec16/17 · plots · cli · slurm/
capture_run.py    headless capture runner → a self-contained .json the widget reloads offline
run_capture.sh    SLURM wrapper · sweep_capture.sh submits a grid · compact_capture.py shrinks captures
run_serve_localhost.sh   serve the GPU backend on SLURM + reverse SSH tunnel
data/             cifar-10-batches-py/ and owt/   (git-ignored; regenerable)
runs_captured/    saved .json captures (git-ignored)
```

Datasets & run outputs are git-ignored — rebuild the OWT shard with `python -m eos_lab.owt_prepare` and
point `--cifar-dir` at the raw CIFAR `data_batch_*` files. See **[eos_lab/README.md](eos_lab/README.md)**
for the package internals.

---

## 📊 Section index (§1–§25)

Every section has a toggle and a caption carrying the full math — this table is just a finder. *Gates:*
**M** = MSE · **CE** = cross-entropy ok · **multi** = needs >1 sample · **N** = small `N` only ·
**GPU** = GPU backend only.

#### Core spectra — *always on*
| § | Panel | Gate |
|---|---|---|
| 1 | loss, sharpness vs `2/η`, residual, spectral edges of `H` | — |
| 2 / 3 | top/bottom-`n` eigenvalues of `H`, loss Hessian, `G`, `S` | — |
| 4 | `J` onto `H` eigvecs in 3 phases (alignment · power-iter · residual-dominance) | — |
| 5 | SLQ spectral density of `H`, `G`, `S`, loss Hessian | — |
| 6 | eigenspace rotation of `H` (principal angles) | — |

#### NTK alignment & residual geometry
| § | Panel | Gate |
|---|---|---|
| 7a / 7 / 8 | NTK alignment + Δλ·Δ⟨r,vₖ⟩ products; multi-sample NTK & FH-SVD; `vec(J)` on FH singular vecs | multi · M/CE |
| 4b / 4c | `J·r` onto `Q[u₁]` eigvecs; `Q[u₁]·(J·r)` onto Gauss–Newton | multi · M/CE |
| 4d | per-residual-sign-group projections | multi · M |

#### σ₁ theory — predicted vs actual sharpness *(the heart of the paper)*
| § | Panel | Gate |
|---|---|---|
| 9 | predicted vs actual σ₁ — Eq. 13 (single), Eq. 21/22/23/29 (multi) | M |
| 9b | same predictions **+ 2nd-order PSD term** `‖ΔJᵀu₁‖²` | M |
| 9c | same predictions vs **full-Hessian** sharpness `λmax(∇²L)` | M |
| 9d / 9d-c | §9 / §9c but residual **self-computed by the quadratic model** (closed loop) | M |
| 10 | **cubic** approximation (Eq. 47/51, ±PSD) vs NTK σ₁ & full-Hessian λmax — *off, heaviest* | M |

#### Per-sample curvature *(small `N`)*
| § | Panel | Gate |
|---|---|---|
| 11 | 3D Hessian–NTK grids `T1/T2/T3` over the `N³` cube | multi · N |
| 12a / 12b | per-sample `Qᵢ=∇²fᵢ`: principal angles + per-sample projections of `∇f` / `J·r` | multi · N |
| 13 / 14 | residual-weighted curvature `≈∑JᵢᵀQⱼJₖrₖ`; `Tr(ΔNTK)` per-triplet decomposition | multi · N |
| 15 | 2nd-difference decomposition of `‖J‖²_F` & σ₁ into theory terms I–VI | M |

#### Curvature-aligned optimizers *(standalone; `M=N·d_out ≤ 256`)*
| § | Panel | Gate |
|---|---|---|
| 16 | project `g₊`/`g₋` onto top/bottom-`kdir` eigvecs of the **averaged** `H̄`; pure-curvature run + 5 baselines | M · small M |
| 17 | per-sample variant — each sample uses its **own** `Qₖ` (`r>0`→top, `r<0`→bottom) | M · small M |

#### Server / browser only (§18–§25)
| § | Panel | Gate |
|---|---|---|
| 18 | §12b per-sample, sample 1 vs sample 2 separately | exactly 2 samples |
| 19 | correlation of Δ‖gradient‖ (first-order) vs `D²(tr NTK)` | M · N |
| 20 | residual-weighted FH `M_r=(1/N)∑rₖQₖ` spectral histograms over training (slider) | N |
| 21 | residual↔spectrum alignment — `r` on NTK, `J·r` on `M_r`, `J·r` on Gauss–Newton (slider) | multi · N |
| 22 / 23 | **quadratic-Taylor REPLACE mode** — frozen-`Q` (§22) or random low-rank `Q` (§23) drives the whole run | M · N |
| 24 | 1st/2nd-order `Δf` alignment time-series (`A=JJᵀr`, `Bₖ` second-order) | GPU · N |
| 25 | gradient-norm & `d/dt(J·r)` evolution (5 curves) | GPU · N |

> **Cost note.** The cheap core (loss / sharpness / eigenvalues / §9 theory / §7a) runs every tick; the
> heavy slowly-varying panels (§7 FH-eigvecs, §8) run every `heavy every` ticks (default 4) and §5 SLQ
> ~50×/run, so a run stays responsive with everything on. Any panel a run can't fill is stamped *"n/a"* or
> *"§X not produced — lower the number of samples"*.

---

## 🎛️ Datasets · architectures · loss · init

Set from the dropdowns; picking a dataset auto-applies a tuned **default preset** (good `lr`/`init`/`N` +
the sections that fit that size). Real-data / conv / transformer runs use the GPU backend.

**Datasets** — `synthetic` (in-browser MLP) · **CIFAR-10** / **MNIST** (10-class) · **CIFAR-2** / **MNIST-2**
(2-class **scalar** ±1 — `d_out=1` keeps the §16/§17 optimizers feasible on real images) · **sorting** (MSE)
· **OpenWebText** (GPT-2-BPE next-token LM; CE or MSE, minibatched) · **Chebyshev** (Cohen et al. EoS toy:
points on [-1,1] labeled by `T_degree`, 6-layer width-50 tanh MLP) · **k-sparse parity** (`ksparse`: parity
of a fixed random `k`-subset of `in_dim` ±1 bits; MLP or mean-pooled mini-GPT) · **angle pair**
(`anglepair`: 2 samples at a controllable angle — pairs with §18/§19) · **saddle** (saddle-to-saddle linear
regression, staircase of escapes) · **agf** (alternating gradient flows, diagonal linear net) · **const**
(constant target — same-sign initial residuals).

**Architectures** — `MLP` · `CNN` · **VGG11** (`chmul` scales channels, up to ~9.4M params) · **mini-GPT**
(sorting regressor / token-LM) · **`diaglin`** (diagonal linear net `β=u⊙v`, used by `agf`).

**Loss** — `MSE` (all panels) or `cross-entropy` (keeps §4/§6; residual-NTK & σ₁-theory panels n/a).
**Init** — `default` (`scale/√fan_in`), `muP`, `Xavier normal/uniform`, or `custom`; `init_scale` sets the
scale/gain/std.

**Two surrogate modes (don't confuse them):**
- **Lockstep compare** (`mode=surrogate`, below the widget) — trains the real model **and** a frozen-window
  Taylor surrogate side-by-side and overlays them (actual solid, surrogate dashed). The linear/NTK
  surrogate diverges once frozen sharpness passes `2/η` — expected at EoS.
- **§22 / §23 REPLACE mode** — the surrogate trajectory *replaces* the real run and drives the core plots
  (§1, §2/§3, §20), with §4–§21/§24 off and a warning banner.

---

## 💾 Batch offline (SLURM / GPU), analyse in the browser later

Run the **full computation headless** (no live server, no tab), save **every section's records** to one
self-contained `.json`, then **load it back into the widget** to scrub all the plots offline — so you can
fan many runs across SLURM/GPUs and analyse them whenever.

```bash
# one capture — flags are plain widget params (s1–s27 + §20/§21 default ON; §16/§17 baselines
# and §22–§25 default OFF, enable per-run via --set sNN=1):
python capture_run.py --dataset cifar10 --arch vgg11 --nsamp 25 --steps 400 \
                      --device cuda:0 --out runs_captured/cifar_vgg.json

# on SLURM (one job → one file); sweep_capture.sh submits a grid:
sbatch --export=ALL,ARGS="--dataset cifar10 --arch mlp --nsamp 25 --steps 400 \
                          --out runs_captured/cifar_mlp.json" run_capture.sh
```

In the widget set **Compute → "⬆ load run"**, pick the `.json`, and it restores the controls and replays
every section into the plots. **"⬇ save run"** downloads the latest **GPU-backend** run (browser-compute
runs aren't record-based). Full details: **[capture_README.md](capture_README.md)**.

---

<sub>🔒 Commits on this box are **local-only by policy** — never pushed.</sub>
