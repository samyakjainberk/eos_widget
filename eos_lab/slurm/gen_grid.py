"""
Generate the EoS/PS run grid — one SLURM bash file per run (à la ../../run4.sh), plus submit_all.sh.

Capped grid: datasets {synthetic(=regression), sorting, cifar10} × architectures {mlp,cnn,vgg11,gpt}
× nsamp {1,10,100,1000,10000} × seeds {0,1}, with nsamp capped ≤1000 for CNN/VGG11. Every run turns
ALL sections on (§1–§9); the multi-sample SVD sections auto-skip when the M×p Jacobian is too large
(handled in diagnostics by `multi_ok`). Each run logs locally to its own hyperparameter-named folder
AND to wandb project 'eos_lab'.

    python eos_lab/slurm/gen_grid.py          # writes eos_lab/slurm/runs/*.sh + submit_all.sh
    bash   eos_lab/slurm/submit_all.sh        # sbatch every run onto the 'main' partition
"""
import os
import stat

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
WIDGET = "/nas/ucb/samsj/TestingPSTheory/eos_widget"
PY = "/nas/ucb/samsj/conda_env/envs/samsenv/bin/python"
CIFAR = "/nas/ucb/samsj/cifar-10-batches-py"   # shared NAS copy (reachable from every compute node)
OUT_ROOT = os.path.join(WIDGET, "eos_lab", "runs")
PROJECT = "eos_lab"

# (dataset, arch): base hyperparameters + per-arch SLURM sizing + allowed nsamp list.
BASES = {
    ("synthetic", "mlp"): dict(  # regression: iid-Gaussian inputs/targets
        sets="act=tanh fixedx=0 depth=4 width=100 indim=10 outdim=1 lr=0.3 init=0.1 steps=800",
        mem="24G", time="12:00:00", nsamps=[1, 10, 100, 1000, 10000]),
    ("sorting", "gpt"): dict(
        sets="dmodel=64 nhead=4 nlayer=2 seqlen=16 lr=0.002 init=0.1 steps=400",
        mem="32G", time="18:00:00", nsamps=[1, 10, 100, 1000, 10000]),
    ("sorting", "mlp"): dict(
        sets="act=tanh depth=4 width=100 seqlen=16 lr=0.05 init=0.3 steps=400",
        mem="24G", time="12:00:00", nsamps=[1, 10, 100, 1000, 10000]),
    ("cifar10", "mlp"): dict(
        sets="loss=mse act=tanh depth=2 width=128 lr=0.02 init=0.5 steps=300",
        mem="32G", time="18:00:00", nsamps=[1, 10, 100, 1000, 10000]),
    ("cifar10", "cnn"): dict(
        sets="loss=mse act=tanh chmul=1.0 lr=0.05 init=0.5 steps=300",
        mem="32G", time="18:00:00", nsamps=[1, 10, 100, 1000]),
    ("cifar10", "vgg11"): dict(
        sets="loss=mse act=tanh chmul=0.25 lr=0.05 init=0.5 steps=300",
        mem="48G", time="24:00:00", nsamps=[1, 10, 100, 1000]),
}
SEEDS = [0, 1]
ALL_SECTIONS = " ".join(f"s{i}=1" for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) + " gson=1 slqprobes=2"


def eig_every(nsamp):
    """Bound the number of diagnostic ticks as N grows (per-step §1–§6 Lanczos gets pricier)."""
    return 2 if nsamp <= 100 else (5 if nsamp < 10000 else 10)


def bash_for(dataset, arch, nsamp, seed, base):
    name = None  # let the CLI build the hyperparameter name; we only set a short job-name
    job = f"eos_{dataset[:4]}_{arch}_n{nsamp}_s{seed}"
    sets = base["sets"] + f" dataset={dataset} arch={arch} nsamp={nsamp} seed={seed} eigevery={eig_every(nsamp)}"
    set_args = " ".join(f"--set {kv}" for kv in (sets.split() + ALL_SECTIONS.split()))
    return f"""#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem={base['mem']}
#SBATCH --gpus=1
#SBATCH --time={base['time']}
#SBATCH --output={HERE}/logs/{job}_%j.out

# WANDB_API_KEY is inherited from the submitting shell (sbatch --export=ALL is the default).
# Fallback: a local, git-ignored key file if you prefer not to export it inline.
[ -z "$WANDB_API_KEY" ] && [ -f "$HOME/.wandb_key" ] && export WANDB_API_KEY="$(cat "$HOME/.wandb_key")"
export WANDB_MODE=online
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

cd {WIDGET}
{PY} -m eos_lab.cli --preset linear --device cuda:0 --cifar-dir {CIFAR} \\
    --wandb --wandb-project {PROJECT} --out-root {OUT_ROOT} \\
    {set_args}
"""


def main():
    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    files = []
    for (dataset, arch), base in BASES.items():
        for nsamp in base["nsamps"]:
            for seed in SEEDS:
                fn = os.path.join(RUNS, f"run_{dataset}_{arch}_n{nsamp}_s{seed}.sh")
                with open(fn, "w") as f:
                    f.write(bash_for(dataset, arch, nsamp, seed, base))
                os.chmod(fn, os.stat(fn).st_mode | stat.S_IEXEC)
                files.append(fn)
    submit = os.path.join(HERE, "submit_all.sh")
    with open(submit, "w") as f:
        f.write("#!/bin/bash\n# sbatch every generated run onto the 'main' partition.\n")
        f.write("# Make sure WANDB_API_KEY is exported first (sbatch propagates it via --export=ALL),\n")
        f.write("# or place it in $HOME/.wandb_key.\nset -e\n")
        f.write('if [ -z "$WANDB_API_KEY" ] && [ ! -f "$HOME/.wandb_key" ]; then\n')
        f.write('  echo "WARNING: WANDB_API_KEY not set and ~/.wandb_key missing — runs will log offline-only."\n')
        f.write("fi\n")
        f.write(f'cd {HERE}\n')
        for fn in files:
            f.write(f"sbatch {os.path.relpath(fn, HERE)}\n")
    os.chmod(submit, os.stat(submit).st_mode | stat.S_IEXEC)
    with open(os.path.join(HERE, ".gitignore"), "w") as f:
        f.write("logs/\nruns/\nsubmit_all.sh\n.wandb_key\n")
    print(f"wrote {len(files)} run scripts to {RUNS}")
    print(f"wrote {submit}")
    print("submit with:  bash " + submit)


if __name__ == "__main__":
    main()
