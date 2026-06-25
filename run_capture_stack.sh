#!/bin/bash
# Run a STACK of headless captures sequentially in one SLURM job (multiple capture_run.py invocations
# back-to-back on the same GPU). The stack file ($STACKFILE) is a plain bash script of capture commands
# written by submit_n125_jobs.sh; each capture writes its own runs_captured/*.json incrementally, so a
# time-limited job still leaves every COMPLETED capture (and a loadable PARTIAL of the running one).
#
#   sbatch --export=ALL,STACKFILE=/abs/path/stack_x.sh run_capture_stack.sh
#
#SBATCH --job-name=eos_stack
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-stack-%j.out

set -u
DIR=/nas/ucb/samsj/TestingPSTheory/eos_widget
cd "$DIR"

echo "================================================================"
echo " EoS capture STACK on node: $(hostname)   job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo " (no GPU visible — CPU)"
echo " Stack file : ${STACKFILE:?set STACKFILE}"
echo " Captures   : $(grep -c capture_run.py "$STACKFILE") runs, sequential"
echo "================================================================"

# Each line in the stack file is a full `python capture_run.py …` command; run them in order.
# On --requeue the whole stack reruns (completed captures are simply overwritten — idempotent).
bash "$STACKFILE"

echo "=== stack complete (job ${SLURM_JOB_ID:-none}) ==="
