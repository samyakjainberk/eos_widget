#!/usr/bin/env python3
"""Compact an EoS widget capture by reducing §14 (g14) records.

Each g14 record stores 45 full N^3 "p14" cubes + 6 "ratio" cubes (~550 KB at N=8),
but §14 has NO cube viewer the browser only reduces every cube to 9 time-series
stats (top-3 / bottom-3 / mean / mean+ / mean-) over the off-diagonal cells
(i!=j!=k) via s14GridStats(), and the ratio cubes to a single off-diagonal mean
via s14RatioMean(). So storing the cubes is pure bloat (g14 = ~81% of a 200 MB
"everything-on" capture).

This rewrites every g14 record to carry ONLY those precomputed stats:
    p14   -> p14r    : p14r[y][g][n] = {top:[3], bot:[3], mean, mpos, mneg}
    ratio -> ratioR  : ratioR[idx]   = off-diagonal mean
index.html's s14GridStats / s14RatioMean read p14r / ratioR directly when present
(see the "compacted capture" fast-paths), so the plots are BIT-IDENTICAL the cube
data was never displayed. Net: g14 shrinks ~60-70x, the file ~6x, and the browser
parse / replay / memory drop accordingly.

Usage:
    python compact_capture.py runs_captured/synth_all.json [more.json ...]
        -> writes runs_captured/synth_all.min.json (non-destructive)
    python compact_capture.py --inplace runs_captured/*_all.json
        -> overwrites each file, moving the original to runs_captured/_full_backup/
"""
import json, os, sys, math, tempfile

def _stats(pk, N):
    """Mirror index.html s14GridStats(): top-3/bot-3 + mean/mpos/mneg over i!=j!=k."""
    if not pk:   # s14GridStats: if(!pk) return all-null stat
        return {"top": [None, None, None], "bot": [None, None, None], "mean": None, "mpos": None, "mneg": None}
    v = pk.get("v") or []
    idx = pk.get("idx")
    N2 = N * N
    off = []
    for q in range(len(v)):
        p = idx[q] if idx is not None else q
        i = p // N2; j = (p % N2) // N; k = p % N
        if i != j and j != k and i != k:
            off.append(v[q])
    # browser: off.sort((a,b)=>b-a)  -> descending; values are finite floats
    off.sort(key=lambda x: (x is None, -(x if isinstance(x, (int, float)) and x == x else float("-inf"))))
    L = len(off)
    s = c = sp = cp = sn = cn = 0.0
    for x in off:
        if x is not None and isinstance(x, (int, float)) and math.isfinite(x):
            s += x; c += 1
            if x > 0: sp += x; cp += 1
            elif x < 0: sn += x; cn += 1
    g = lambda a: (a if a is not None else None)
    return {"top": [g(off[0]) if L > 0 else None, g(off[1]) if L > 1 else None, g(off[2]) if L > 2 else None],
            "bot": [g(off[L - 1]) if L > 0 else None, g(off[L - 2]) if L > 1 else None, g(off[L - 3]) if L > 2 else None],
            "mean": (s / c if c else None), "mpos": (sp / cp if cp else None), "mneg": (sn / cn if cn else None)}

def _ratio_mean(pk, N):
    """Mirror index.html s14RatioMean(): off-diagonal (i!=j!=k) mean of one ratio cube."""
    if not pk:   # s14RatioMean: if(!snap.ratio[idx]) return null
        return None
    v = pk.get("v") or []
    idx = pk.get("idx")
    N2 = N * N
    s = c = 0.0
    for q in range(len(v)):
        p = idx[q] if idx is not None else q
        i = p // N2; j = (p % N2) // N; k = p % N; x = v[q]
        if i != j and j != k and i != k and x is not None and isinstance(x, (int, float)) and math.isfinite(x):
            s += x; c += 1
    return (s / c if c else None)

def compact_record(r):
    """Reduce one g14 record in place. Returns (changed, bytes_before_est)."""
    if r.get("type") != "g14":
        return False
    N = r.get("M")
    if not N:
        return False
    if "p14" in r:
        p14r = {}
        for y, groups in r["p14"].items():
            p14r[y] = {g: [_stats(pk, N) for pk in cubes] for g, cubes in groups.items()}
        r["p14r"] = p14r
        del r["p14"]
    if "ratio" in r:
        r["ratioR"] = [_ratio_mean(pk, N) for pk in r["ratio"]]
        del r["ratio"]
    return True

def compact_file(path, inplace=False):
    with open(path) as f:
        obj = json.load(f)
    recs = obj.get("records", [])
    n14 = sum(compact_record(r) for r in recs)
    obj["compacted"] = True
    # write atomically
    out = path if inplace else (os.path.splitext(path)[0] + ".min.json")
    d = os.path.dirname(os.path.abspath(out)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    if inplace:
        bak = os.path.join(d, "_full_backup")
        os.makedirs(bak, exist_ok=True)
        os.replace(path, os.path.join(bak, os.path.basename(path)))
    os.replace(tmp, out)
    return out, n14

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    inplace = "--inplace" in sys.argv
    if not args:
        print(__doc__); sys.exit(1)
    for path in args:
        b0 = os.path.getsize(path)
        out, n14 = compact_file(path, inplace=inplace)
        b1 = os.path.getsize(out)
        print("%-40s %6.1f MB -> %6.1f MB  (%4.1fx)  [%d g14 records]%s"
              % (os.path.basename(path), b0 / 1e6, b1 / 1e6, b0 / max(b1, 1), n14,
                 "  [in-place; original -> _full_backup/]" if inplace else "  -> " + os.path.basename(out)))
