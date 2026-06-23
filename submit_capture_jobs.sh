#!/bin/bash
# Submit a stack of "EVERYTHING ON" headless capture jobs to SLURM — every section in the on-position
# EXCEPT §8 (s8), §4b (s9), §4c (s10). Each writes one self-contained runs_captured/*.json you load later
# in the widget (Compute → "⬆ load run"). Captures save incrementally (~60s), so a time-limited job still
# leaves a loadable PARTIAL.
#
#   ./submit_capture_jobs.sh
#
# Design (so a 2-3 hr all-sections run stays loadable in a browser):
#  • eigevery=1               — §15 panels 1-4 (the per-step 2nd-difference) need 3 CONSECUTIVE GD steps.
#  • cubeevery=8              — the §11-§14 N³ Hessian-NTK cubes are the file-size driver (~0.5 MB/step at N=8);
#                              emit them every 8th tick. §15 / §1-§9 / §16-§17 stay FULL resolution. ⇒ ~0.07 MB/step.
#  • small N (≤8)             — the cubes are N³, so N must stay small to stay loadable.
#  • §16/§17 on with baselines (s24base/s25base=1, kdir=32) but bounded s24iter — they run UPFRONT.

set -u
TIME=${TIME:-04:00:00}
SUB="sbatch --time=$TIME --parsable"
mkdir -p runs_captured

# EVERYTHING on except §8/§4b/§4c; cube cadence 8; §16/§17 baselines on but bounded.
ALL="--set s8=0 --set s9=0 --set s10=0 --set cubeevery=8 --set s24base=1 --set s25base=1 --set s24iter=25"

declare -a JOBS=(
  "synth_all|--dataset synthetic --arch mlp --width 64 --act tanh --set fixedx=0 --nsamp 8 --lr 0.3 --steps 2500 $ALL"
  "cheby_all|--dataset chebyshev --arch mlp --depth 6 --width 50 --nsamp 8 --lr 0.1 --steps 2500 $ALL"
  "const_all|--dataset const --arch mlp --width 64 --act tanh --set fixedx=0 --nsamp 8 --lr 0.3 --steps 2000 $ALL"
  "ksparse_all|--dataset ksparse --arch mlp --indim 12 --set ksparse=4 --nsamp 8 --width 64 --lr 0.05 --steps 2500 $ALL"
  "cifar2_all|--dataset cifar2 --arch mlp --width 64 --nsamp 8 --lr 0.02 --steps 2000 $ALL"
  "mnist2_all|--dataset mnist2 --arch mlp --width 48 --nsamp 8 --lr 0.02 --steps 2000 $ALL"
  "cifar10_all|--dataset cifar10 --arch mlp --width 64 --nsamp 6 --lr 0.02 --steps 1500 $ALL"
  "mnist_all|--dataset mnist --arch mlp --width 48 --nsamp 6 --lr 0.02 --steps 1500 $ALL"
  "ksparse_gpt_all|--dataset ksparse --arch gpt --indim 12 --set ksparse=4 --nsamp 8 --dmodel 48 --lr 0.005 --steps 700 $ALL"
)

echo "submitting ${#JOBS[@]} EVERYTHING-ON capture jobs (except §8/§4b/§4c; time=$TIME each) → runs_captured/"
for entry in "${JOBS[@]}"; do
  name="${entry%%|*}"; args="${entry#*|}"
  full="$args --label $name --out runs_captured/${name}.json"
  jid=$($SUB --job-name="cap_${name}" --export=ALL,ARGS="$full" run_capture.sh)
  echo "  $name  → job $jid"
done
echo "done. Watch:  squeue -u $USER | grep cap_   ·   results in runs_captured/*.json (load each via Compute → '⬆ load run')."
