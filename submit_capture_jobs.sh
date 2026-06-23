#!/bin/bash
# Submit a stack of headless capture jobs to SLURM. Each writes one self-contained runs_captured/*.json
# you load later in the widget (Compute → "⬆ load run"). Captures save incrementally (every ~60s), so a
# job killed by the time-limit still leaves a loadable PARTIAL.
#
#   ./submit_capture_jobs.sh
#
# Design notes:
#  • eigevery=1 everywhere — §15 panels 1-4 (the per-step 2nd-difference) need 3 CONSECUTIVE GD steps.
#  • The §11-§14 3D-grid/cube sections (s18/s19/s20/s21/s22) are OFF in the long runs — §14's N³ triplet
#    cubes are ~8 MB/step at N=20, which would make a multi-thousand-step capture gigabytes. One dedicated
#    small-N "cubes" job keeps them.
#  • §16/§17 (s24/s25) run UPFRONT before the main loop, so they're OFF in the EoS runs (which want many
#    §1-§15 steps) and ON only in the short, dedicated optimizer jobs.

set -u
TIME=${TIME:-04:00:00}                       # wall limit; incremental save means a kill still yields a partial
SUB="sbatch --time=$TIME --parsable"
mkdir -p runs_captured

# common: turn OFF the heavy 3D-cube sections (§11-§14) and §16/§17 for the EoS/§15 runs
EOS="--set s18=0 --set s19=0 --set s20=0 --set s21=0 --set s22=0 --set s24=0 --set s25=0"
# optimizer runs: §16/§17 ON with their 5 baselines, cubes OFF, few main steps (the optimizer runs upfront)
OPT="--set s18=0 --set s19=0 --set s20=0 --set s21=0 --set s22=0 --set s24=1 --set s25=1 --set s24base=1 --set s25base=1 --set s24iter=80"

declare -a JOBS=(
  # ---- EoS + §15 (many steps, §1-§9 + §15) ----
  "cheby_eos|--dataset chebyshev --arch mlp --depth 6 --width 50 --nsamp 20 --lr 0.1 --steps 3000 $EOS"
  "synth_eos|--dataset synthetic --arch mlp --width 64 --act tanh --set fixedx=0 --nsamp 24 --lr 0.3 --steps 2000 $EOS"
  "ksparse_eos|--dataset ksparse --arch mlp --indim 12 --set ksparse=4 --nsamp 24 --width 64 --lr 0.05 --steps 1500 $EOS"
  "cifar2_eos|--dataset cifar2 --arch mlp --width 64 --nsamp 24 --lr 0.02 --steps 1200 $EOS"
  "mnist2_eos|--dataset mnist2 --arch mlp --width 48 --nsamp 24 --lr 0.02 --steps 1200 $EOS"
  "cifar10_eos|--dataset cifar10 --arch mlp --width 64 --nsamp 20 --lr 0.02 --steps 800 $EOS"
  # ---- §16/§17 curvature optimizer (kdir=32, 5 baselines) + a short §1-§15 tail ----
  "ksparse_opt|--dataset ksparse --arch mlp --indim 12 --set ksparse=4 --nsamp 24 --lr 0.05 --steps 60 $OPT"
  "cifar2_opt|--dataset cifar2 --arch mlp --nsamp 16 --lr 0.02 --steps 60 $OPT"
  # ---- one dedicated small-N job that KEEPS the §11-§14 3D cubes (default on) ----
  "synth_cubes|--dataset synthetic --arch mlp --width 48 --act tanh --set fixedx=0 --nsamp 8 --lr 0.3 --steps 250 --set s24=0 --set s25=0"
)

echo "submitting ${#JOBS[@]} capture jobs (time=$TIME each) → runs_captured/"
for entry in "${JOBS[@]}"; do
  name="${entry%%|*}"; args="${entry#*|}"
  full="$args --label $name --out runs_captured/${name}.json"
  jid=$($SUB --job-name="cap_${name}" --export=ALL,ARGS="$full" run_capture.sh)
  echo "  $name  → job $jid"
done
echo "done. Watch:  squeue -u $USER --name=eos_capture   ·   results land in runs_captured/*.json"
echo "load each .json in the widget (Compute → '⬆ load run')."
