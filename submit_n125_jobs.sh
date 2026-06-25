#!/bin/bash
# Submit "full-batch GD on N=125 samples" capture jobs — EVERY section EXCEPT the user's 6 and the three
# N³-cube sections that cannot be loaded at N=125.
#
#   OFF (user):   §8 §4b §4c §16 §17 §18                       = s8 s9 s10 s24 s25 s26
#   OFF (N³,unloadable at N=125): §11 §13 §14                  = s18 s21 s20
#       (§11/§13/§14 are 3D grids/cubes over all N³=1.95M sample-TRIPLES → ~2M-point scatter & multi-GB/snapshot;
#        the browser can't render or load that. §12a/§12b are only N²=15,625 sample-PAIRS, so they DO run.)
#   ON (everything else): §1 §2 §3 §4 §4d §5 §6 §7 §7a §9 §9c §9d §9d-c §10 §12a §12b §15 §19 §20
#
# These are SLOW at N=125 (per-sample Lanczos, frozen-Hessian SVD, cubic) — that's expected & accepted.
# cubeevery=8 throttles the heavy §12a/§12b/§20 snapshots so files stay loadable; §1-§10/§15/§19 stay per-tick.
# Captures save incrementally (~60s) so a time-limited job still leaves a loadable PARTIAL. Each SLURM file
# STACKS several captures (sequential on one GPU). Load each via the widget's Compute → "⬆ load from server".
#
#   ./submit_n125_jobs.sh
#
set -u
DIR=/nas/ucb/samsj/TestingPSTheory/eos_widget
PY=/nas/ucb/samsj/conda_env/envs/samsenv/bin/python
CIFAR="$DIR/data/cifar-10-batches-py"
SDIR="$DIR/n125_stacks"
TIME=${TIME:-40:00:00}
mkdir -p "$SDIR" "$DIR/runs_captured"

# Full-batch GD on 125 samples + every section ON except the user's 6 and §11/§13/§14 (N³ cubes).
COMMON="--nsamp 125 --batch 0 --optimizer gd --loss mse --outdim 1 --tgt 1.0 --init 0.5 --bias 1 \
 --set grid3dcap=130 --set sec12ncap=130 --set cubeevery=8 \
 --set s8=0 --set s9=0 --set s10=0 --set s24=0 --set s25=0 --set s26=0 \
 --set s18=0 --set s20=0 --set s21=0"

# jobs: "stackid|name|per-run args".  small-p datasets get more steps (faster); large-p (cifar/mnist) fewer
# steps + eigevery=1 still, relying on incremental partials.  eigevery=1 so §15's per-step 2nd-difference fills.
JOBS=(
  "cif2|cifar2_lr01|--dataset cifar2 --arch mlp --width 64 --depth 2 --lr 0.01 --steps 200 --eigevery 1"
  "cif2|cifar2_lr02|--dataset cifar2 --arch mlp --width 64 --depth 2 --lr 0.02 --steps 200 --eigevery 1"
  "cif5|cifar2_lr05|--dataset cifar2 --arch mlp --width 64 --depth 2 --lr 0.05 --steps 200 --eigevery 1"
  "cif10|cifar10_lr01|--dataset cifar10 --arch mlp --width 64 --depth 2 --lr 0.01 --steps 200 --eigevery 1"
  "cif10|cifar10_lr02|--dataset cifar10 --arch mlp --width 64 --depth 2 --lr 0.02 --steps 200 --eigevery 1"
  "mn2|mnist2_lr02|--dataset mnist2 --arch mlp --width 64 --depth 2 --lr 0.02 --steps 400 --eigevery 1"
  "mn2|mnist2_lr05|--dataset mnist2 --arch mlp --width 64 --depth 2 --lr 0.05 --steps 400 --eigevery 1"
  "mn|mnist_lr02|--dataset mnist --arch mlp --width 64 --depth 2 --lr 0.02 --steps 400 --eigevery 1"
  "mn|mnist_lr05|--dataset mnist --arch mlp --width 64 --depth 2 --lr 0.05 --steps 400 --eigevery 1"
  "ksA|ksparse_lr02|--dataset ksparse --arch mlp --indim 12 --set ksparse=4 --width 64 --depth 2 --lr 0.02 --steps 1000 --eigevery 1"
  "ksA|ksparse_lr05|--dataset ksparse --arch mlp --indim 12 --set ksparse=4 --width 64 --depth 2 --lr 0.05 --steps 1000 --eigevery 1"
  "ksB|ksparse_lr1|--dataset ksparse --arch mlp --indim 12 --set ksparse=4 --width 64 --depth 2 --lr 0.1 --steps 1000 --eigevery 1"
  "ksB|ksparse_lr2|--dataset ksparse --arch mlp --indim 12 --set ksparse=4 --width 64 --depth 2 --lr 0.2 --steps 1000 --eigevery 1"
  "chb|cheby_lr05|--dataset chebyshev --arch mlp --depth 4 --width 50 --lr 0.05 --steps 1000 --eigevery 1"
  "chb|cheby_lr1|--dataset chebyshev --arch mlp --depth 4 --width 50 --lr 0.1 --steps 1000 --eigevery 1"
  "syn|synth_lr3|--dataset synthetic --arch mlp --width 64 --depth 2 --lr 0.3 --steps 1000 --eigevery 1"
  "syn|synth_lr5|--dataset synthetic --arch mlp --width 64 --depth 2 --lr 0.5 --steps 1000 --eigevery 1"
)

declare -A STACKS; ORDER=()
for entry in "${JOBS[@]}"; do
  sid="${entry%%|*}"; rest="${entry#*|}"; name="${rest%%|*}"; args="${rest#*|}"
  cmd="\"$PY\" -u capture_run.py $COMMON $args --label $name --out runs_captured/n125_${name}.json --cifar-dir \"$CIFAR\" --device auto || echo \"[stack] FAILED: $name\""
  if [ -z "${STACKS[$sid]+x}" ]; then ORDER+=("$sid"); fi
  STACKS[$sid]+="echo \"--- [$name] \$(date) ---\""$'\n'"$cmd"$'\n'
done

echo "submitting ${#ORDER[@]} stacked SLURM jobs (${#JOBS[@]} captures, time=$TIME each) → runs_captured/"
for sid in "${ORDER[@]}"; do
  sf="$SDIR/stack_${sid}.sh"
  { echo "#!/bin/bash"; echo "set -u"; echo "cd $DIR"; printf '%s' "${STACKS[$sid]}"; } > "$sf"
  chmod +x "$sf"
  n=$(grep -c capture_run.py "$sf")
  jid=$(sbatch --time="$TIME" --parsable --job-name="n125_$sid" --export=ALL,STACKFILE="$sf" run_capture_stack.sh)
  echo "  stack $sid : $n captures → job $jid"
done
echo "done. Watch: squeue -u $USER | grep n125_   ·   results: runs_captured/n125_*.json"
