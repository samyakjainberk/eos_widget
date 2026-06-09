#!/bin/bash
# Launch the EoS / Progressive-Sharpening GPU backend on an A100 node via SLURM AND open a
# reverse SSH tunnel back to the login host (rnn) so the widget is reachable at
# http://localhost:8756/ from anywhere that has rnn's localhost forwarded (e.g. the VSCode
# remote session, which auto-forwards to your laptop).
#
# Node-to-node HTTP is firewall-blocked and /home is node-local, so we cannot tunnel rnn->node.
# Instead the node dials BACK to rnn (compute->login SSH is open) with a dedicated, port-forward-only
# key on the shared NAS, binding rnn:127.0.0.1:8756 -> this node's localhost:8756.
#
#   sbatch run_a100_localhost.sh        # submit
#   scancel <jobid>                     # stop (also tears down the tunnel)
#
#SBATCH --job-name=eos_localhost
#SBATCH --partition=main
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=40:00:00
#SBATCH --output=slurm-%j.out

set -u
PORT=${PORT:-8756}   # override per-instance:  sbatch --export=ALL,PORT=8757 run_a100_localhost.sh
LOGIN=rnn.ist.berkeley.edu
PY=/nas/ucb/samsj/conda_env/envs/samsenv/bin/python
DIR=/nas/ucb/samsj/TestingPSTheory/eos_widget
CIFAR="$DIR/data/cifar-10-batches-py"   # in-repo copy (shared NAS → readable from compute nodes)
KEY=/nas/ucb/samsj/.eos_tunnel/id_tunnel
KH=/nas/ucb/samsj/.eos_tunnel/known_hosts

echo "================================================================"
echo " EoS GPU backend + reverse tunnel on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo " Server  : http://0.0.0.0:$PORT/  (this node)"
echo " Tunnel  : ${LOGIN}:127.0.0.1:$PORT  ->  $(hostname):localhost:$PORT"
echo " Open    : http://localhost:$PORT/   (Compute -> 'GPU backend')"
echo "================================================================"

cd "$DIR"

# 1) Start the GPU backend in the background.
"$PY" -u server.py --devices all --port "$PORT" --host 0.0.0.0 --cifar-dir "$CIFAR" &
SERVER_PID=$!

cleanup() { echo "[launcher] cleaning up"; kill "$SERVER_PID" 2>/dev/null; kill "$SSH_PID" 2>/dev/null; }
trap cleanup EXIT TERM INT

# Wait for the server to bind before opening the tunnel.
for i in $(seq 1 60); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/"; then echo "[launcher] server up"; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "[launcher] server died early"; exit 1; fi
  sleep 1
done

# 2) Keep a reverse tunnel to rnn alive for as long as the server runs.
while kill -0 "$SERVER_PID" 2>/dev/null; do
  ssh -i "$KEY" -N -T \
      -o UserKnownHostsFile="$KH" \
      -o StrictHostKeyChecking=yes \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -R 127.0.0.1:$PORT:localhost:$PORT \
      samsj@"$LOGIN" &
  SSH_PID=$!
  wait "$SSH_PID"
  echo "[launcher] tunnel dropped (rc=$?); retrying in 5s"
  sleep 5
done

echo "[launcher] server exited; shutting down"
