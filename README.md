# TriView-TA Framework

Source code for **TriView-TA**, a transform-aware tri-view framework for robust multivariate time-series classification under structured spectral shift.

The corresponding manuscript is:

**Transform-Aware Cross-Frequency Consistency for Robust Multivariate Time-Series Classification under Spectral Shift**

## What is included

- `src/`: dataset loaders, spectral transforms, model components, losses, and training entry points.
- `scripts/`: experiment scripts for benchmark runs, Safe-A calibration, perturbation evaluation, sensitivity analyses, complexity profiling, and reviewer-requested diagnostic experiments.
- `tools/`: visualization and post-hoc analysis utilities, including tri-view examples, cross-view geometry evolution, and confusion-matrix diagnostics.
- `tests/`: lightweight checks for dataset and transform logic.
- `requirements.txt`: Python dependencies used by the released implementation.

Data files, checkpoints, generated figures, and experiment outputs are intentionally not tracked in this repository.

## Main benchmark datasets

The seven datasets used in the main benchmark of the manuscript are:

- `UWaveGestureLibrary`
- `JapaneseVowels`
- `SpokenArabicDigits`
- `Handwriting`
- `FaceDetection`
- `Heartbeat`
- `HHAR`

For the six standard UEA/UCR-style benchmark datasets, the implementation follows the official train/test split when available and uses a validation split from the training data for model selection. For HHAR, the manuscript reports the WOODS-style real-world OOD protocol based on device-model environments.

## Installation

```bash
pip install -r requirements.txt
```

A CUDA-enabled PyTorch environment is recommended for reproducing the main experiments.

## Basic training command

The following command gives a representative TriView-TA run on a UEA-style dataset:

```bash
python src/train_uea.py \
  --dataset UWaveGestureLibrary \
  --device cuda \
  --seed 42 \
  --dataset-profile auto \
  --supervised-views triview \
  --backbone all \
  --triview-fusion gated \
  --use-temporal-attn \
  --use-shared-qk-attn \
  --shared-qk-heads 4 \
  --pretrain-epochs 0 \
  --epochs 20 \
  --finetune-epochs 20 \
  --batch-size 32 \
  --num-workers 0 \
  --eval-num-workers 0
```

The option `--dataset-profile auto` applies dataset-level defaults such as STFT configuration and validation split settings. In the manuscript, STFT preprocessing is treated as a dataset-level protocol component: all methods that use externally constructed frequency-domain or time-frequency inputs share the same STFT setting within each dataset.

## Example commands for the main benchmark datasets

```bash
python src/train_uea.py --dataset UWaveGestureLibrary --device cuda --seed 42 --dataset-profile auto --supervised-views triview --backbone all --triview-fusion gated --use-temporal-attn --use-shared-qk-attn --shared-qk-heads 4 --pretrain-epochs 0 --epochs 20 --finetune-epochs 20 --batch-size 32 --num-workers 0 --eval-num-workers 0

python src/train_uea.py --dataset JapaneseVowels --device cuda --seed 42 --dataset-profile auto --supervised-views triview --backbone all --triview-fusion gated --use-temporal-attn --use-shared-qk-attn --shared-qk-heads 4 --pretrain-epochs 0 --epochs 20 --finetune-epochs 20 --batch-size 32 --num-workers 0 --eval-num-workers 0

python src/train_uea.py --dataset SpokenArabicDigits --device cuda --seed 42 --dataset-profile auto --supervised-views triview --backbone all --triview-fusion gated --use-temporal-attn --use-shared-qk-attn --shared-qk-heads 4 --pretrain-epochs 0 --epochs 20 --finetune-epochs 20 --batch-size 32 --num-workers 0 --eval-num-workers 0

python src/train_uea.py --dataset Handwriting --device cuda --seed 42 --dataset-profile auto --supervised-views triview --backbone all --triview-fusion gated --use-temporal-attn --use-shared-qk-attn --shared-qk-heads 4 --pretrain-epochs 0 --epochs 20 --finetune-epochs 20 --batch-size 32 --num-workers 0 --eval-num-workers 0

python src/train_uea.py --dataset FaceDetection --device cuda --seed 42 --dataset-profile auto --supervised-views triview --backbone all --triview-fusion gated --use-temporal-attn --use-shared-qk-attn --shared-qk-heads 4 --pretrain-epochs 0 --epochs 20 --finetune-epochs 20 --batch-size 32 --num-workers 0 --eval-num-workers 0

python src/train_uea.py --dataset Heartbeat --device cuda --seed 42 --dataset-profile auto --supervised-views triview --backbone all --triview-fusion gated --use-temporal-attn --use-shared-qk-attn --shared-qk-heads 4 --pretrain-epochs 0 --epochs 20 --finetune-epochs 20 --batch-size 32 --num-workers 0 --eval-num-workers 0
```

## HHAR preprocessing and OOD evaluation

HHAR requires preprocessing before the device-based OOD experiment.

```bash
python scripts/preprocess_hhar.py
```

The manuscript uses a WOODS-style HHAR protocol where environments are defined by device model. Source devices are used for training/validation, and one device is held out as the unseen target environment. Users should check local dataset paths before running HHAR experiments.

## Safe-A calibration and perturbation evaluation

The Safe-A protocol is used to retain approximately semantics-preserving perturbation severities based on reference-model prediction agreement.

Relevant scripts include:

```bash
python scripts/teacher_agreement.py
python scripts/ref_rule_sensitivity.py
python scripts/safe_signal.py
python scripts/sweep_transforms.py
```

These scripts support reference-model agreement analysis, reference-rule sensitivity, signal-level safety diagnostics, and perturbation sweeps.

## Sensitivity and diagnostic analyses

The repository includes scripts for the main diagnostic analyses reported in the manuscript:

```bash
python scripts/run_dg_baselines_table2.py
python scripts/run_loss_weight_sensitivity.py
python scripts/run_stft_sensitivity.py
python scripts/profile_complexity_six.py
python scripts/eval_multiview_coordination.py
python scripts/eval_repr_consistency.py
python scripts/eval_transform_recovery.py
```

Visualization and post-hoc diagnostic utilities are provided in `tools/`:

```bash
python tools/plot_triview_examples.py
python tools/collect_cv_evolution.py
python tools/plot_cv_evolution.py
python tools/plot_cv_evolution_paper_aligned.py
python tools/select_and_plot_confusion_mixed.py
python tools/plot_hhar_confusion_mixed.py
```

These tools correspond to the manuscript analyses on tri-view construction examples, cross-view geometry evolution, perturbation-induced drift, and confusion-matrix diagnostics.

## Minimal recipe for applying TriView-TA to a new dataset

1. Prepare a multivariate time-series classification dataset with train/test splits.
2. Hold out a validation subset from the training set for model selection and Safe-A calibration.
3. Estimate normalization statistics only from the training/source split.
4. Construct three deterministic views:
   - time view: raw time series;
   - frequency view: temporally pooled STFT magnitude;
   - time-frequency view: log-compressed STFT magnitude.
5. Select STFT parameters according to sequence length and temporal resolution.
6. Train TriView-TA with the default loss weights used in the manuscript.
7. Calibrate Safe-A on the training/validation split using reference-model prediction agreement.
8. Report clean accuracy/Macro-F1, worst-case score in the retained safe region, average score in the retained safe region, and clean-to-worst degradation.
9. Report efficiency information, including parameter count, MACs, batch inference latency, hardware, and batch size.

For long-sequence or resource-constrained deployment, users may consider sliding-window or chunked STFT processing, window-level prediction aggregation, disabling optional attention-related components, branch sharing, or distilling the tri-view model into a compact student model.

## Tests

```bash
pip install pytest
pytest -q
```

## Notes

This repository is intended to support reproducibility and further research on transform-aware robust time-series classification. Exact numerical reproduction may depend on local dataset preprocessing, hardware, CUDA/PyTorch versions, random seeds, and dataset availability.
