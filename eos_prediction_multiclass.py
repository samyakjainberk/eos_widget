#!/usr/bin/env python
"""Headless capture runner for the MULTICLASS PREDICTION widget
(eos_widget_prediction/index_prediction_multiclass.html).

Identical flow to eos_prediction.py (run_stream predictions + Prediction-1&2 sweep, merged into one
"⬆ load run" file) but with the multiclass defaults: a real d-class dataset + cross-entropy loss.

Datasets (multiclass, d=10 by default): maxfind (d-class argmax — synthetic, no data files, the default
so it "just runs"), modadd ((a+b) mod m — synthetic), and cifar10 / mnist (need --cifar-dir). The
explicit-Jacobian predictions need N·outD ≤ grid3dcap (500), so at d=10 keep nsamp ≤ 50.

  # default: maxfind (10-class argmax), CE loss, small MLP — every prediction panel produces:
  python eos_prediction_multiclass.py --out runs_captured/mc_maxfind.json

  # (a+b) mod 11, transformer:
  python eos_prediction_multiclass.py --dataset modadd --arch gpt --outdim 11 \
                                      --out runs_captured/mc_modadd.json

  # CIFAR-10, 10-class MLP (needs the raw batches):
  python eos_prediction_multiclass.py --dataset cifar10 --cifar-dir /path/to/cifar-10-batches-py \
                                      --nsamp 25 --out runs_captured/mc_cifar.json

NOTE: the Prediction-1&2 sweep's ‖J·ṙ‖−‖J̇·r‖ theory is squared-loss only, so under the default CE loss
run_sweep is skipped (the capture records the predictions and notes "0 sweep points — MSE-only"). To
also capture the sweep, run a small MSE config, e.g. `--loss mse --dataset maxfind --nsamp 8`.
"""
import eos_prediction


def multiclass_defaults():
    """prediction_defaults() retargeted to a multiclass CE dataset."""
    p = eos_prediction.prediction_defaults()
    p["dataset"] = "maxfind"    # d-class argmax finder — synthetic, CE-natural, no data files (fast, always produces)
    p["arch"] = "mlp"
    p["loss"] = "ce"            # cross-entropy: the p×p Gauss-Newton spectrum becomes the Fisher (see the CE notes)
    p["indim"] = "10"           # maxfind: inD = outD = indim = #classes (d)
    p["outdim"] = "10"          # (used by modadd/cifar10/mnist; harmless for maxfind)
    p["nsamp"] = "10"           # N·outD = 100 ≤ grid3dcap (500) ⇒ predictions run
    return p


def main():
    eos_prediction.run_capture(multiclass_defaults, title="prediction_multiclass")


if __name__ == "__main__":
    main()
