#!/bin/bash
# Run eos_lab jobs across A100 GPUs via a SLURM array — one job per (preset/seed), each on its own GPU.
#
#   sbatch --array=0-3 eos_lab/slurm/submit_array.sh        # 4 jobs (indices map to RUNS below)
#   squeue -u $USER ; tail -f eos_lab/slurm/slurm-<jobid>_<idx>.out
#
# Each array task gets 1 GPU (--gres ...:1); SLURM sets CUDA_VISIBLE_DEVICES so cuda:0 = that GPU.
#
#SBATCH --job-name=eos_lab
#SBATCH --partition=main
#SBATCH --gres=gpu:A100-SXM4-80GB:1          # or gpu:A100-PCI-80GB:1 / gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=eos_lab/slurm/slurm-%A_%a.out

set -e
PY=/nas/ucb/samsj/conda_env/envs/samsenv/bin/python
DIR=/nas/ucb/samsj/TestingPSTheory/eos_widget
CIFAR="$DIR/data/cifar-10-batches-py"   # in-repo copy on shared NAS (readable from compute nodes; /home is node-local)

# Edit this list; one entry per array index. Each is "preset key=val key=val ...".
RUNS=(
  "msample seed=0 steps=800"
  "msample seed=1 steps=800"
  "cifar_mlp nsamp=128 steps=300"
  "cifar_vgg chmul=0.25 nsamp=64 steps=300"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
SPEC=${RUNS[$IDX]}
read -r PRESET RESTSETS <<< "$SPEC"
SET_ARGS=""
for kv in $RESTSETS; do SET_ARGS="$SET_ARGS --set $kv"; done

echo "=========================================================="
echo " eos_lab array task $IDX on $(hostname): preset=$PRESET sets=[$RESTSETS]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================================="

cd "$DIR"
exec "$PY" -m eos_lab.cli --preset "$PRESET" --device cuda:0 --cifar-dir "$CIFAR" \
     --out "runs/${PRESET}_idx${IDX}" $SET_ARGS
