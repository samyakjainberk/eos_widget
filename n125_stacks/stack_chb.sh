#!/bin/bash
set -u
cd /nas/ucb/samsj/TestingPSTheory/eos_widget
echo "--- [cheby_lr05] $(date) ---"
"/nas/ucb/samsj/conda_env/envs/samsenv/bin/python" -u capture_run.py --nsamp 125 --batch 0 --optimizer gd --loss mse --outdim 1 --tgt 1.0 --init 0.5 --bias 1  --set grid3dcap=130 --set sec12ncap=130 --set cubeevery=8  --set s8=0 --set s9=0 --set s10=0 --set s24=0 --set s25=0 --set s26=0  --set s18=0 --set s20=0 --set s21=0 --dataset chebyshev --arch mlp --depth 4 --width 50 --lr 0.05 --steps 1000 --eigevery 1 --label cheby_lr05 --out runs_captured/n125_cheby_lr05.json --cifar-dir "/nas/ucb/samsj/TestingPSTheory/eos_widget/data/cifar-10-batches-py" --device auto || echo "[stack] FAILED: cheby_lr05"
echo "--- [cheby_lr1] $(date) ---"
"/nas/ucb/samsj/conda_env/envs/samsenv/bin/python" -u capture_run.py --nsamp 125 --batch 0 --optimizer gd --loss mse --outdim 1 --tgt 1.0 --init 0.5 --bias 1  --set grid3dcap=130 --set sec12ncap=130 --set cubeevery=8  --set s8=0 --set s9=0 --set s10=0 --set s24=0 --set s25=0 --set s26=0  --set s18=0 --set s20=0 --set s21=0 --dataset chebyshev --arch mlp --depth 4 --width 50 --lr 0.1 --steps 1000 --eigevery 1 --label cheby_lr1 --out runs_captured/n125_cheby_lr1.json --cifar-dir "/nas/ucb/samsj/TestingPSTheory/eos_widget/data/cifar-10-batches-py" --device auto || echo "[stack] FAILED: cheby_lr1"
