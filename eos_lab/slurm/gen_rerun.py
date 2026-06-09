"""
Generate the *feasible* re-run scripts for the multi-sample sections (§4/4b/4c/4d/7/8/9-multi).

Context: those sections form the explicit M×p Jacobian (M = N·oc) and were auto-skipped whenever M
exceeded the old diagnostics budget (max_M=512).  The budget was since raised to max_M=2048, which
admits the M≈1000–1600 runs (every n100 plus synthetic mlp n1000) but still blocks the genuinely
infeasible n1000+ runs (M≥10000 → 16-160 GB Jacobian and a 20k-iteration §8 loop per tick).

The one remaining cost is §8's O(M) finite-difference loop (2·M backward passes per tick).  At the
original eigevery=2 that is ~10× the per-tick cost of the M=100 runs (which already took 4–8 h), so
each feasible re-run raises eigevery to keep `M · ticks` near the M=100 baseline — same wall-clock,
coarser temporal sampling.  Expensive models (vgg11/gpt) get a larger bump for headroom.

These re-run with the *fixed* code (per-sample H eigenvalues, repaired §9 plot), so their output is
correct directly — no post-hoc rerender needed.  Each overwrites its run dir only on success, so the
already-fixed results.pt is preserved if a job times out.

    python eos_lab/slurm/gen_rerun.py     # writes eos_lab/slurm/rerun/*.sh + submit_rerun.sh
    bash   eos_lab/slurm/rerun/submit_rerun.sh
"""
import os
import re
import stat

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "runs")                 # the as-launched run scripts (source of truth)
OUT = os.path.join(HERE, "rerun")

# (source script stem) -> tuned eigevery.  M and per-model cost drive the choice (see module docstring).
FEASIBLE = {
    "run_cifar10_mlp_n100_s0":    15, "run_cifar10_mlp_n100_s1":    15,   # M=1000, cheap mlp
    "run_cifar10_cnn_n100_s0":    20, "run_cifar10_cnn_n100_s1":    20,   # M=1000, medium cnn
    "run_cifar10_vgg11_n100_s0":  40, "run_cifar10_vgg11_n100_s1":  40,   # M=1000, expensive vgg
    "run_sorting_gpt_n100_s0":    40, "run_sorting_gpt_n100_s1":    40,   # M=1600, expensive gpt
    "run_sorting_mlp_n100_s0":    20, "run_sorting_mlp_n100_s1":    20,   # M=1600, cheap mlp
    "run_synthetic_mlp_n1000_s0": 20, "run_synthetic_mlp_n1000_s1": 20,   # M=1000, tiny mlp
}


def main():
    os.makedirs(OUT, exist_ok=True)
    written = []
    for stem, ee in FEASIBLE.items():
        src = os.path.join(SRC, stem + ".sh")
        if not os.path.exists(src):
            print(f"  WARN missing source {src} — skipped")
            continue
        txt = open(src).read()
        txt = re.sub(r"--set eigevery=\d+", f"--set eigevery={ee}", txt)
        txt = re.sub(r"(#SBATCH --job-name=\S+)", r"\1_rr", txt)          # mark in squeue
        txt = re.sub(r"(#SBATCH --output=)(.*?)_%j\.out", r"\1\2_rr_%j.out", txt)
        # vgg11's big p + conv activations + the §9 fh_frozen Jacobians need >16 GB at M=1000 — pin it
        # to a 48 GB A6000 so it can't land on a 16 GB A4000 node and OOM.
        if "vgg11" in stem:
            txt = txt.replace("#SBATCH --gpus=1", "#SBATCH --gres=gpu:A6000:1")
        dst = os.path.join(OUT, stem + "_rr.sh")
        with open(dst, "w") as f:
            f.write(txt)
        os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC)
        written.append(dst)
    submit = os.path.join(OUT, "submit_rerun.sh")
    with open(submit, "w") as f:
        f.write("#!/bin/bash\n# Re-submit the feasible multi-sample re-runs (raised-budget gate).\nset -e\n")
        f.write(f"cd {OUT}\n")
        for d in written:
            f.write(f"sbatch {os.path.basename(d)}\n")
    os.chmod(submit, os.stat(submit).st_mode | stat.S_IEXEC)
    print(f"wrote {len(written)} re-run scripts to {OUT}")
    print(f"submit with:  bash {submit}")


if __name__ == "__main__":
    main()
