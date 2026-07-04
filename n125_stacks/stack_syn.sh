#!/bin/bash
set -u
cd /nas/ucb/samsj/TestingPSTheory/eos_widget
echo "--- [synth_lr3] $(date) ---"
"/nas/ucb/samsj/conda_env/envs/samsenv/bin/python" -u capture_run.py --nsamp 125 --batch 0 --optimizer gd --loss mse --outdim 1 --tgt 1.0 --init 0.5 --bias 1  --set grid3dcap=130 --set sec12ncap=130 --set cubeevery=8  --set s8=0 --set s9=0 --set s10=0 --set s24=0 --set s25=0 --set s26=0  --set s18=0 --set s20=0 --set s21=0 --dataset synthetic --arch mlp --width 64 --depth 2 --lr 0.3 --steps 1000 --eigevery 1 --label synth_lr3 --out runs_captured/n125_synth_lr3.json --cifar-dir "/nas/ucb/samsj/TestingPSTheory/eos_widget/data/cifar-10-batches-py" --device auto || echo "[stack] FAILED: synth_lr3"
echo "--- [synth_lr5] $(date) ---"
"/nas/ucb/samsj/conda_env/envs/samsenv/bin/python" -u capture_run.py --nsamp 125 --batch 0 --optimizer gd --loss mse --outdim 1 --tgt 1.0 --init 0.5 --bias 1  --set grid3dcap=130 --set sec12ncap=130 --set cubeevery=8  --set s8=0 --set s9=0 --set s10=0 --set s24=0 --set s25=0 --set s26=0  --set s18=0 --set s20=0 --set s21=0 --dataset synthetic --arch mlp --width 64 --depth 2 --lr 0.5 --steps 1000 --eigevery 1 --label synth_lr5 --out runs_captured/n125_synth_lr5.json --cifar-dir "/nas/ucb/samsj/TestingPSTheory/eos_widget/data/cifar-10-batches-py" --device auto || echo "[stack] FAILED: synth_lr5"
