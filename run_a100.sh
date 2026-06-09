#!/bin/bash
# Launch the EoS / Progressive-Sharpening GPU backend on an A100 (80 GB) node via SLURM.
# The A100s have ~3-4x the memory of the local A6000s and are much faster — use them for the big
# CIFAR-VGG11 / mini-GPT runs and the surrogate-comparison sections.
#
#   sbatch run_a100.sh            # submit; watch with:  squeue -u $USER   /  tail -f slurm-<jobid>.out
#   scancel <jobid>               # stop it
#
# Then, FROM YOUR LAPTOP, tunnel to the allocated node (its name is printed in slurm-<jobid>.out):
#   ssh -N -L 8756:<a100-node>:8756 <you>@rnn.ist.berkeley.edu      # forward through the login host
#   open http://localhost:8756/                                     # set Compute -> "GPU backend"
#
# For an INTERACTIVE session instead of a batch job:
#   srun --partition=main --gres=gpu:A100-SXM4-80GB:1 --cpus-per-task=8 --mem=64G --time=8:00:00 --pty bash
#   then on the node: /nas/ucb/samsj/conda_env/envs/samsenv/bin/python server.py \
#        --device cuda:0 --port 8756 --host 0.0.0.0 --cifar-dir /home/alexandreduplessis/.torch/cifar-10-batches-py
#
#SBATCH --job-name=eos_a100
#SBATCH --partition=main                       # A100 nodes: airl/sac (SXM4-80GB), cirl/rlhf (PCI-80GB). Use 'scavenger' if main is full (preemptible).
#SBATCH --gres=gpu:A100-SXM4-80GB:2            # 2 GPUs → up to 2 concurrent browser tabs, each on its own GPU (or gpu:A100-PCI-80GB:2)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -e
PORT=8756
PY=/nas/ucb/samsj/conda_env/envs/samsenv/bin/python
DIR=/nas/ucb/samsj/TestingPSTheory/eos_widget
CIFAR=/home/alexandreduplessis/.torch/cifar-10-batches-py

echo "================================================================"
echo " EoS GPU backend on A100 node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo " Serving port $PORT (host 0.0.0.0)."
echo " From your laptop, tunnel over SSH to THIS node (node-to-node HTTP is blocked, so forward"
echo " the node's localhost via an SSH ProxyJump through the login host):"
echo "   ssh -N -L $PORT:localhost:$PORT -J <you>@rnn.ist.berkeley.edu <you>@$(hostname)"
echo "   (or directly:  ssh -N -L $PORT:localhost:$PORT <you>@$(hostname)  if the node is reachable)"
echo "   open http://localhost:$PORT/     (Compute -> 'GPU backend')"
echo "================================================================"

cd "$DIR"
# --devices all → pool every GPU SLURM gave this job (2 here). Concurrent browser tabs are auto-assigned
# the least-busy GPU, so two simultaneous runs land on different GPUs. Use --device cuda:0 for single-GPU.
exec "$PY" -u server.py --devices all --port "$PORT" --host 0.0.0.0 --cifar-dir "$CIFAR"
