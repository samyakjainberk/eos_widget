#!/bin/bash
# Generate + submit a grid of STACKED prediction-widget captures to SLURM.
# Grid = datasets × learning-rates × init-schemes, ALL plots on (Pred-3/4/5/6 + ray + early-dynamics +
# 2c/2d for MSE datasets). Each SLURM job runs PERJOB captures sequentially (~12–15 h). Load each .json
# in the multiclass widget via "⬆ load run" when done.
#
#   ./gen_pred_sweep.sh                 # submit the default grid
#   DRYRUN=1 ./gen_pred_sweep.sh        # print the manifests + sbatch lines, submit NOTHING
#   STEPS=1500 PERJOB=5 ./gen_pred_sweep.sh
set -u
DIR=/nas/ucb/samsj/TestingPSTheory/eos_widget
cd "$DIR"; mkdir -p runs_captured/manifests runs_captured/logs

# ------------------------------- EDIT THE GRID HERE -------------------------------
# dataset:loss:arch   (cifar10/mnist use MSE ⇒ the sweep + 2c/2d produce; maxfind/modadd use CE)
DATASETS=( "cifar10:mse:mlp" "mnist:mse:mlp" "maxfind:ce:mlp" "modadd:ce:mlp" )
LRS=( 0.02 0.05 0.12 )                          # learning rates (spread to span below→through EoS)
INITS=( default kaiming_normal xavier_normal )  # init schemes (widget dropdown options)
WIDTH=${WIDTH:-32}
STEPS=${STEPS:-1200}                            # GD steps per capture (long ⇒ full PS/EoS story)
SWPAIRS=${SWPAIRS:-120}                         # Prediction-1&2 sweep pairs (MSE datasets)
SWSTEPS=${SWSTEPS:-100}
SEED=${SEED:-0}
PERJOB=${PERJOB:-4}                             # captures stacked per SLURM job (tune to ~12–15 h)
DRYRUN=${DRYRUN:-0}
# ----------------------------------------------------------------------------------

ALL=$(mktemp)
for d in "${DATASETS[@]}"; do
  IFS=: read -r ds loss arch <<< "$d"
  case "$ds" in
    modadd) NS=22; DIMS="--indim 22 --outdim 11" ;;   # (a+b) mod 11: 2m in, m out; N·outD = 242 ≤ grid3dcap 500
    *)      NS=20; DIMS="--indim 10 --outdim 10" ;;    # d=10 classes; N·d = 200 ≤ 500 ⇒ predictions produce
  esac
  script="eos_prediction_multiclass.py"
  for lr in "${LRS[@]}"; do
    for sc in "${INITS[@]}"; do
      name="${ds}_${arch}_${loss}_lr${lr}_${sc}_n${NS}_s${SEED}"
      out="runs_captured/${name}.json"
      echo "$script --dataset $ds --loss $loss --arch $arch --nsamp $NS $DIMS --lr $lr --initscheme $sc --width $WIDTH --steps $STEPS --swpairs $SWPAIRS --swsteps $SWSTEPS --seed $SEED --label $name --out $out" >> "$ALL"
    done
  done
done

total=$(wc -l < "$ALL")
rm -f runs_captured/manifests/job_*.txt
split -l "$PERJOB" -d --additional-suffix=.txt "$ALL" runs_captured/manifests/job_
njobs=$(ls runs_captured/manifests/job_*.txt | wc -l)
echo "grid: $total captures → $njobs stacked jobs ($PERJOB per job)"

for m in runs_captured/manifests/job_*.txt; do
  jn="eosp_$(basename "$m" .txt)"
  if [ "$DRYRUN" = "1" ]; then
    echo "---- $jn ($m) ----"; sed 's/^/     /' "$m"
    echo "     sbatch --job-name=$jn --export=ALL,MANIFEST=$DIR/$m run_capture_pred_stack.sh"
  else
    sbatch --job-name="$jn" --export=ALL,MANIFEST="$DIR/$m" run_capture_pred_stack.sh
  fi
done
rm -f "$ALL"
[ "$DRYRUN" = "1" ] && echo "(DRYRUN — submitted nothing)" || echo "submitted $njobs jobs → runs_captured/ (load each .json in the widget)"
