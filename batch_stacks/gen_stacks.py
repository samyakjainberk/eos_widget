#!/usr/bin/env python3
"""Generate SLURM capture-stack files for the requested EoS runs.

Runs 1,2  : single-sample synthetic, tanh, lr=0.1, init=0.4, 100-wide 4-layer MLP.
            r1 = positive initial residual, r2 = negative (ssign).
            sections §1,§2,§3,§4 only.
Run 3     : paired-sample anglepair (indim=100), lr=0.1, 100-wide 4-layer MLP.
            sections §1,§2,§3,§4,§18 (so §12 proj s19/s22 + §18 s26 on).
            grid: {tanh init0.4, quadratic init1.0} x {++,+-,--} x {30,90,150 deg}.
All runs  : 2000 steps, full-batch GD, bias off, scalar output, seed 0.
"""
import os

PY  = "/nas/ucb/samsj/conda_env/envs/samsenv/bin/python"
DIR = "/nas/ucb/samsj/TestingPSTheory/eos_widget"
OUT = "runs_captured"

# section toggles: capture_run defaults EVERYTHING on, so we turn off all but what we want.
ALL = list(range(1, 29))
def offsets(keep):
    return " ".join("--set s%d=%d" % (s, 1 if s in keep else 0) for s in ALL)

KEEP_12   = {1, 2, 3, 4}                 # runs 1,2: §1-§4
KEEP_18   = {1, 2, 3, 4, 19, 22, 26}     # run 3: §1-§4 + §12(proj) + §18
OFF12 = offsets(KEEP_12)
OFF3  = offsets(KEEP_18)

COMMON = "--arch mlp --width 100 --depth 4 --outdim 1 --bias 0 --lr 0.1 --steps 2000 --seed 0"

runs = []   # (filename_stem, full capture_run.py arg string)

# ---- runs 1 & 2 : single-sample synthetic ----
for stem, ssign in [("single_tanh_posResid", "pos"), ("single_tanh_negResid", "neg")]:
    args = ("%s --dataset synthetic --act tanh --indim 10 --nsamp 1 --init 0.4 --eigevery 1 "
            "--ssign %s %s --label %s --out %s/%s.json"
            % (COMMON, ssign, OFF12, stem, OUT, stem))
    runs.append((stem, args))

# ---- run 3 : paired anglepair grid ----
signcombos = [("pp", 1, 1), ("pm", 1, -1), ("mm", -1, -1)]
angles = [30, 90, 150]
acts = [("tanh", "tanh", 0.4), ("quad", "quadratic", 1.0)]   # (tag, act-name, init)
for atag, aname, init in acts:
    for sc, l1, l2 in signcombos:
        for ang in angles:
            stem = "pair_%s_%s_ang%d" % (atag, sc, ang)
            args = ("%s --dataset anglepair --act %s --indim 100 --nsamp 2 --init %s --eigevery 2 "
                    "--set lab1=%d --set lab2=%d --set angle=%d %s --label %s --out %s/%s.json"
                    % (COMMON, aname, init, l1, l2, ang, OFF3, stem, OUT, stem))
            runs.append((stem, args))

print("total runs:", len(runs))

# ---- distribute into 5 stacks (round-robin so heavy run-3 jobs spread evenly) ----
NSTACKS = 5
stacks = [[] for _ in range(NSTACKS)]
for i, r in enumerate(runs):
    stacks[i % NSTACKS].append(r)

os.makedirs(os.path.join(DIR, "batch_stacks"), exist_ok=True)
paths = []
for k, stk in enumerate(stacks):
    p = os.path.join(DIR, "batch_stacks", "stack_%d.sh" % (k + 1))
    with open(p, "w") as f:
        f.write("#!/bin/bash\n# auto-generated capture stack %d (no set -e: one failing/diverging capture must not abort the rest)\ncd %s\n" % (k + 1, DIR))
        for stem, args in stk:
            f.write('echo "==== capture: %s ===="\n' % stem)
            f.write("%s -u capture_run.py %s\n" % (PY, args))
        f.write('echo "==== stack %d done ===="\n' % (k + 1))
    paths.append(p)
    print("stack_%d.sh : %d runs -> %s" % (k + 1, len(stk), ", ".join(s for s, _ in stk)))

print("\nstack files written:")
for p in paths:
    print(" ", p)
