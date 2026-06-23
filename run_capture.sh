#!/bin/bash
# Headless EoS / Progressive-Sharpening capture on a SLURM GPU node.
#
# Runs capture_run.py (the SAME compute as the GPU backend's /run, every §1–§17 toggle on by default)
# and writes a single self-contained .json you load later in the widget (Compute → "⬆ load run") to
# analyse the plots offline — no live server, no tab kept open. Fan many of these out in parallel.
#
#   # one run (everything on, defaults):
#   sbatch --export=ALL,ARGS="--out runs_captured/demo.json" run_capture.sh
#
#   # a real config — ARGS are plain capture_run.py flags:
#   sbatch --export=ALL,ARGS="--dataset cifar10 --arch vgg11 --nsamp 25 --steps 400 \
#                             --out runs_captured/cifar_vgg.json" run_capture.sh
#
#   # a sweep: submit many, each its own job/file (see sweep_capture.sh for a grid helper):
#   for lr in 0.02 0.05 0.1; do
#     sbatch --export=ALL,ARGS="--dataset cifar10 --arch mlp --nsamp 25 --steps 400 \
#            --lr $lr --out runs_captured/cifar_mlp_lr${lr}.json" run_capture.sh
#   done
#
#SBATCH --job-name=eos_capture
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-capture-%j.out

set -u
PY=${PY:-/nas/ucb/samsj/conda_env/envs/samsenv/bin/python}
DIR=/nas/ucb/samsj/TestingPSTheory/eos_widget
CIFAR="$DIR/data/cifar-10-batches-py"   # in-repo copy (shared NAS → readable from compute nodes)
ARGS=${ARGS:-"--out runs_captured/capture-${SLURM_JOB_ID:-local}.json"}

cd "$DIR"
mkdir -p runs_captured

echo "================================================================"
echo " EoS headless capture on node: $(hostname)   job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo " (no GPU visible — will use CPU)"
echo " ARGS : $ARGS"
echo "================================================================"

# default to the in-repo CIFAR dir unless the caller already passed --cifar-dir
case "$ARGS" in
  *--cifar-dir*) EXTRA="" ;;
  *)             EXTRA="--cifar-dir $CIFAR" ;;
esac
# default device auto (GPU if present); capture_run.py picks float32 on GPU, float64 on CPU
case "$ARGS" in
  *--device*) ;;
  *) EXTRA="$EXTRA --device auto" ;;
esac

echo "[run_capture] $PY capture_run.py $ARGS $EXTRA"
exec "$PY" -u capture_run.py $ARGS $EXTRA
