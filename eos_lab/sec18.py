"""§18 — per-sample (sample-1 vs sample-2) projection time-series — STANDALONE re-slice of §12b.

§18 is NOT a separate compute: the server emits no `g18` record. The §18 per-sample data is the
`pbysamp`/`jbysamp`/`vbysamp` positional arrays already produced inside `sec12_payload`
(eos_lab/linalg.py rank_stats; server.py `_sec12_payload`) and shipped in the §12 `g4d` record.
The browser (index.html `s18Append`/`S18_PHASES`) renders §18 by SLICING those arrays at sample
index 0 vs 1, for 3 phases × 2 samples (6 single-row plots). §18 only renders at N=2 (`nsamp=2`).

This module mirrors that browser slice EXACTLY:

  S18_PHASES = [
    ('vbysamp', 'vals',  'phase 1: alignment',          '|⟨J,u⟩|/‖J‖'),   # cosine alignment; old captures fall back to masked vals
    ('jbysamp',  None,   'phase 2: power iteration',     '|⟨J,u⟩|·σ'),     # eigenvalue-scaled projection (new field — no fallback)
    ('pbysamp', 'pvals', 'phase 3: residual dominance',  '|⟨J,u⟩|·σ·r'),   # §12b product; old captures fall back to masked pvals
  ]

For each phase the browser reads `m.proj.top[rk]` and `m.proj.bot[rk]` (rk=0,1 = rank-1 / rank-2
eigenvectors of each per-sample Hessian Q_i), preferring the positional `(v|j|p)bysamp` field and
falling back to the histogram-masked `vals`/`pvals` for phases 1 and 3 only (phase 2 `jbysamp` has
NO fallback — null where absent). It then picks element `[si]` for sample si ∈ {0,1}; a value that
is None or non-finite becomes a gap (null).

So the §18 record is a pure VIEW of the §12 payload: no new linear algebra, no new HVPs. The
authoritative numerical content lives in the §12 `proj` arrays (which already match server
1e-13 in fp64 on the multi-sample primitives). The `sec18_payload` below produces a small,
self-describing record (the same per-(phase, sample, group, rank) scalars the browser plots) so the
integration phase can ship a thin `g18` view-record if desired, or so it can be parity-checked
directly against the server's §12 arrays.

INTEGRATION NOTE: §18 has no standalone driver. To emit it, diagnostics.py must run the §12 payload
(widen the §12 trigger + emit to include `s26`) so the `g4d` record carries proj.top/bot; then call
`sec18_payload(g4d)` (only when `s26` and N==2) to produce the optional `g18` view. See PORT SPEC §3b.
"""

# (phase field, fallback field, phase title, y-axis label) — MIRRORS index.html S18_PHASES exactly.
S18_PHASES = [
    ("vbysamp", "vals",  "phase 1: alignment",         "|⟨J,u⟩|/‖J‖"),
    ("jbysamp", None,    "phase 2: power iteration",    "|⟨J,u⟩|·σ"),
    ("pbysamp", "pvals", "phase 3: residual dominance", "|⟨J,u⟩|·σ·r"),
]

# group → (rank-1 colour, rank-2 colour): top (blue) / bottom (red). MIRRORS index.html S18_GRP2.
S18_GROUPS = ["top", "bot"]
S18_NRANK = 2   # rank-1 (solid) + rank-2 (dotted) eigenvectors per group


def _isfinite(x):
    # mirror browser `isFinite(arr[si])` — None / NaN / ±Inf all become a gap (null)
    return x is not None and x == x and x not in (float("inf"), float("-inf"))


def _pick(ranks, rk, key, fb, si):
    """One §18 scalar = the browser's `s18Append` inner pick:
        e = ranks[rk]; arr = e ? (e[key] || (fb ? e[fb] : null)) : null;
        value = (arr && arr.length>si && arr[si]!=null && isFinite(arr[si])) ? arr[si] : null
    Prefer the positional bysamp field; fall back to the masked histogram list (phases 1 & 3 only);
    return None (a plotted gap) when the rank/field/sample is absent, spurious, or non-finite.
    NB the masked fallback (`vals`/`pvals`) is positionally MISALIGNED w.r.t. true sample index
    (it drops spurious / ‖J‖=0 samples), so it is a best-effort legacy fallback — the eos_lab/server
    payloads always populate the positional bysamp arrays, so the fallback is only hit for old captures."""
    e = ranks[rk] if (ranks is not None and rk < len(ranks)) else None
    if e is None:
        return None
    arr = e.get(key)
    if not arr:                                  # missing OR empty → try fallback (phases 1 & 3 only)
        arr = e.get(fb) if fb is not None else None
    if not arr or si >= len(arr):
        return None
    v = arr[si]
    return float(v) if _isfinite(v) else None


def sec18_payload(sec12):
    """Re-slice the §12 payload (`g4d`, the dict returned by linalg.sec12_payload) into the §18
    per-sample view. Renders/returns only at N=2 (the proj arrays must carry ≥2 samples).

    Returns a dict mirroring the 6 §18 plots (3 phases × 2 samples), each carrying the 4 traces
    (top-rank1, top-rank2, bot-rank1, bot-rank2):

      {"M": 2,
       "phases": [{"key","title","ylabel"}, …(3)],
       "samples": [0, 1],
       "plots": [ { "phase": ph, "sample": si, "title": "<phase> — sample <si+1>",
                    "ylabel": "…",
                    "top": [r1, r2],   # rank-1, rank-2 (None = gap)
                    "bot": [r1, r2] },
                  …(6, ordered ph*2+si like index.html `p_s18_<ph*2+si>`) ] }

    None where a rank-2 eigenvector is absent, the eigvec is spurious, ‖J‖=0, or (phase 2) the
    capture predates `jbysamp`. This is a pure VIEW: the numbers come straight from `sec12.proj`."""
    if sec12 is None:
        return None
    proj = sec12.get("proj") if isinstance(sec12, dict) else None
    M = sec12.get("M") if isinstance(sec12, dict) else None
    if proj is None or proj.get("top") is None or M != 2:
        return None   # §18 is N=2 only; nothing to slice otherwise (matches browser `m.M!==2` guard)

    plots = []
    for ph, (key, fb, title, ylabel) in enumerate(S18_PHASES):
        for si in range(2):                       # sample 0 / sample 1
            rec = {"phase": ph, "sample": si,
                   "title": f"{title} — sample {si + 1}", "ylabel": ylabel}
            for grp in S18_GROUPS:
                ranks = proj.get(grp) or []
                rec[grp] = [_pick(ranks, rk, key, fb, si) for rk in range(S18_NRANK)]
            plots.append(rec)

    return {"M": 2,
            "phases": [{"key": k, "title": t, "ylabel": y} for (k, _fb, t, y) in S18_PHASES],
            "samples": [0, 1],
            "plots": plots}
