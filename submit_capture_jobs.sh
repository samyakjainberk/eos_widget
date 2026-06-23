#!/bin/bash
# Submit a stack of "EVERYTHING ON" headless capture jobs to SLURM — every section in the on-position
# EXCEPT §8 (s8), §4b (s9), §4c (s10). Each writes one self-contained runs_captured/*.json you load later
# in the widget (Compute → "⬆ load run"). Captures save incrementally (~60s) → a time-limited job still
# leaves a loadable PARTIAL.
#
#   ./submit_capture_jobs.sh
#
# Why the config is what it is (so a 2-3 hr all-sections run actually produces every panel AND loads in a browser):
#  • eigevery=1        — §15 panels 1-4 (the per-step 2nd-difference) need 3 CONSECUTIVE GD steps.
#  • cubeevery=8       — the §11-§14 N³ Hessian-NTK cubes are the file-size driver; emit every 8th tick.
#                        §15 / §1-§9 / §16-§17 stay FULL resolution. ⇒ ~0.07 MB/step (vs 0.57) at N=8.
#  • small N (≤8)      — the cubes are N³, so N must stay small to stay loadable.
#  • §16/§17 run UPFRONT, before §1-§15. Their per-sample BASELINES (×6 trajectories) at high kdir are so slow
#    they'd eat the whole job and §1-§15 would never run. So the EVERYTHING-ON jobs run §16/§17 MAIN only
#    (baselines off, kdir=8 ⇒ fast) — the §16/§17 panels still populate. The §16/§17 dotted BASELINES (the
#    shuffle/random/gd comparison) are captured by the SEPARATE *_opt jobs (kdir=32) below.

set -u
TIME=${TIME:-04:00:00}
SUB="sbatch --time=$TIME --parsable"
mkdir -p runs_captured

# EVERYTHING on except §8/§4b/§4c; cube cadence 8; §16/§17 MAIN only (fast) so §1-§15 always run.
EVERY="--set s8=0 --set s9=0 --set s10=0 --set cubeevery=8 --set s24base=0 --set s25base=0 --set s24k=8 --set s24iter=20"
# Dedicated §16/§17 optimizer + the 5 dotted baselines (kdir=32), §1-§15 off so it runs fully.
OPT="--set cubeevery=8 --set s24base=1 --set s25base=1 --set s24k=32 --set s24iter=40 \
 --set s1=0 --set s2=0 --set s3=0 --set s4=0 --set s5=0 --set s6=0 --set s7=0 --set s8=0 --set s9=0 --set s10=0 \
 --set s11=0 --set s12=0 --set s13=0 --set s14=0 --set s15=0 --set s16=0 --set s17=0 \
 --set s18=0 --set s19=0 --set s20=0 --set s21=0 --set s22=0 --set s23=0 --set s24=1 --set s25=1"

declare -a JOBS=(
  # ---- EVERYTHING ON (except §8/§4b/§4c) — §1-§15 + cubes + §15 + §16/§17 main ----
  "synth_all|--dataset synthetic --arch mlp --width 64 --act tanh --set fixedx=0 --nsamp 8 --lr 0.3 --steps 2500 $EVERY"
  "cheby_all|--dataset chebyshev --arch mlp --depth 6 --width 50 --nsamp 8 --lr 0.1 --steps 2500 $EVERY"
  "const_all|--dataset const --arch mlp --width 64 --act tanh --set fixedx=0 --nsamp 8 --lr 0.3 --steps 2000 $EVERY"
  "ksparse_all|--dataset ksparse --arch mlp --indim 12 --set ksparse=4 --nsamp 8 --width 64 --lr 0.05 --steps 2500 $EVERY"
  "cifar2_all|--dataset cifar2 --arch mlp --width 64 --nsamp 8 --lr 0.02 --steps 2000 $EVERY"
  "mnist2_all|--dataset mnist2 --arch mlp --width 48 --nsamp 8 --lr 0.02 --steps 2000 $EVERY"
  "cifar10_all|--dataset cifar10 --arch mlp --width 64 --nsamp 6 --lr 0.02 --steps 1500 $EVERY"
  "mnist_all|--dataset mnist --arch mlp --width 48 --nsamp 6 --lr 0.02 --steps 1500 $EVERY"
  "ksparse_gpt_all|--dataset ksparse --arch gpt --indim 12 --set ksparse=4 --nsamp 8 --dmodel 48 --lr 0.005 --steps 700 $EVERY"
  # ---- dedicated §16/§17 + 5 baselines (kdir=32), the shuffle-vs-main optimizer study ----
  "ksparse_opt|--dataset ksparse --arch mlp --indim 12 --set ksparse=4 --nsamp 8 --lr 0.05 --steps 1 $OPT"
  "cifar2_opt|--dataset cifar2 --arch mlp --nsamp 8 --lr 0.02 --steps 1 $OPT"
)

echo "submitting ${#JOBS[@]} jobs (time=$TIME each) → runs_captured/"
for entry in "${JOBS[@]}"; do
  name="${entry%%|*}"; args="${entry#*|}"
  full="$args --label $name --out runs_captured/${name}.json"
  jid=$($SUB --job-name="cap_${name}" --export=ALL,ARGS="$full" run_capture.sh)
  echo "  $name  → job $jid"
done
echo "done. Watch:  squeue -u $USER | grep cap_   ·   results in runs_captured/*.json (load each via Compute → '⬆ load run')."
