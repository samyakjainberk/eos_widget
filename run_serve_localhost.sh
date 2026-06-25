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
#SBATCH --job-name=eos_serve
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=40:00:00
#SBATCH --requeue                 # auto-requeue on node failure / preemption (don't lose the server)
#SBATCH --open-mode=append        # keep the same log across a requeue
#SBATCH --output=slurm-%j.out

set -u
PORT=${PORT:-8756}      # override per-instance:  sbatch --export=ALL,PORT=8757 run_serve_localhost.sh
DEVICES=${DEVICES:-cpu} # cpu (float64, no GPU gres) OR all/cuda (float32, needs --gres=gpu:1):
                        #   GPU: sbatch --gres=gpu:1 --export=ALL,PORT=8756,DEVICES=all run_serve_localhost.sh
                        #   CPU: sbatch          --export=ALL,PORT=8758,DEVICES=cpu run_serve_localhost.sh
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

SERVER_PID=""; SSH_PID=""

start_server() {
  "$PY" -u server.py --devices "$DEVICES" --port "$PORT" --host 0.0.0.0 --cifar-dir "$CIFAR" &
  SERVER_PID=$!
  echo "[launcher] server started (pid $SERVER_PID)"
}
start_tunnel() {
  ssh -i "$KEY" -N -T \
      -o UserKnownHostsFile="$KH" \
      -o StrictHostKeyChecking=yes \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -R 127.0.0.1:$PORT:localhost:$PORT \
      samsj@"$LOGIN" &
  SSH_PID=$!
  echo "[launcher] tunnel started (pid $SSH_PID)"
}
wait_bind() {   # block until the server answers on $PORT (or it dies → restart and keep waiting)
  for _ in $(seq 1 120); do
    curl -s -o /dev/null "http://127.0.0.1:$PORT/" && { echo "[launcher] server up"; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "[launcher] server died during startup — restarting"; sleep 3; start_server; }
    sleep 1
  done
}

cleanup() { echo "[launcher] cleaning up"; kill "$SERVER_PID" "$SSH_PID" 2>/dev/null; }
trap cleanup EXIT TERM INT

# 1) Start the GPU backend, wait for it to bind, then 2) open the reverse tunnel.
start_server
wait_bind
start_tunnel

# 3) SUPERVISOR: keep BOTH alive for the whole allocation. If the server crashes (bad config, transient
#    CUDA error, OOM) or the SSH tunnel drops, restart just that one — the job/connection never goes down
#    on its own. Only a SLURM time-limit / scancel / node-failure ends it, and --requeue brings it back.
while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[launcher] server process gone — restarting in 3s"
    sleep 3; start_server; wait_bind
  fi
  if ! kill -0 "$SSH_PID" 2>/dev/null; then
    echo "[launcher] tunnel down — reconnecting"
    start_tunnel
  fi
  sleep 10
done
