#!/bin/bash
# Headless PREDICTION-widget captures, STACKED — one SLURM job runs many captures sequentially so each
# job fills ~12–15 h of GPU time. Each line of the manifest is a full eos_prediction*.py invocation
# (script + flags); this wrapper runs them one after another, continuing past any single failure, and
# writes one self-contained .json (+ .min.json) per capture that you load in the widget via "⬆ load run".
#
#   sbatch --export=ALL,MANIFEST=/abs/path/to/job_00.txt run_capture_pred_stack.sh
#
# Generate the manifests + submit the whole grid with:  ./gen_pred_sweep.sh
#
#SBATCH --job-name=eos_pred_stack
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=runs_captured/logs/stack-%j.out

set -u
PY=${PY:-/nas/ucb/samsj/conda_env/envs/samsenv/bin/python}
DIR=/nas/ucb/samsj/TestingPSTheory/eos_widget
CIFAR="$DIR/data/cifar-10-batches-py"          # in-repo CIFAR/MNIST batches (shared NAS → readable on compute nodes)
MANIFEST=${MANIFEST:?set MANIFEST=/abs/path/to/manifest.txt (one eos_prediction*.py command per line)}

cd "$DIR"
mkdir -p runs_captured/logs

echo "================================================================"
echo " eos_pred_stack on node: $(hostname)   job=${SLURM_JOB_ID:-local}"
echo " manifest: $MANIFEST   ($(grep -cvE '^\s*(#|$)' "$MANIFEST") captures)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo " (no GPU visible — CPU)"
echo "================================================================"

n=0; ok=0; t_all=$(date +%s)
while IFS= read -r line; do
  case "$line" in ''|\#*) continue ;; esac                     # skip blank / comment lines
  n=$((n+1))
  extra=""
  case "$line" in *--cifar-dir*) ;; *cifar10*|*mnist*) extra="--cifar-dir $CIFAR" ;; esac   # auto CIFAR/MNIST dir
  case "$line" in *--device*) ;; *) extra="$extra --device auto" ;; esac                     # GPU on the node (fp32)
  echo ">>>> [$n] $(date '+%F %T')  $PY -u $line $extra"
  t0=$(date +%s)
  if $PY -u $line $extra; then ok=$((ok+1)); st="ok"; else st="FAILED (continuing)"; fi
  echo "<<<< [$n] $st   ($(( $(date +%s) - t0 ))s)"
done < "$MANIFEST"

echo "================================================================"
echo " stack done: $ok/$n captures ok   total $(( ($(date +%s) - t_all) / 60 )) min"
echo "================================================================"
