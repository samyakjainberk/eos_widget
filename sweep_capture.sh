#!/bin/bash
# Submit a small grid of headless captures to SLURM — one job (one GPU, one .json) per config.
# Edit the arrays below, then:  ./sweep_capture.sh
# Each capture is loadable in the widget via Compute → "⬆ load run".

set -u
OUTDIR=${OUTDIR:-runs_captured}
mkdir -p "$OUTDIR"

# ---- edit your sweep here ----
DATASETS=(cifar10 mnist)
ARCHS=(mlp vgg11)
LRS=(0.02 0.05)
NSAMP=${NSAMP:-25}
STEPS=${STEPS:-400}
# ------------------------------

n=0
for ds in "${DATASETS[@]}"; do
  for arch in "${ARCHS[@]}"; do
    for lr in "${LRS[@]}"; do
      name="${ds}_${arch}_lr${lr}_n${NSAMP}"
      out="$OUTDIR/${name}.json"
      args="--dataset $ds --arch $arch --nsamp $NSAMP --steps $STEPS --lr $lr --label $name --out $out"
      echo "submit: $name"
      sbatch --job-name="cap_${name}" --export=ALL,ARGS="$args" run_capture.sh
      n=$((n+1))
    done
  done
done
echo "submitted $n capture jobs → $OUTDIR/ (load each .json in the widget when done)"
