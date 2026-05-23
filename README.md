# time

Minimal MVP for spectral-shift robustness experiments in time series.

## What is included
- Dataset that returns multi-view dict (time, frequency, time-frequency, shift/scale/color variants) for rapid experimentation.
- Transform operators for frequency scaling, band translation, and spectral coloring.
- Losses for TF-Consistency and TA-CFC (VICReg/InfoNCE).
- Minimal training/validation/test loop with logging.

## Quick start
```bash
python src/train.py --epochs 3 --batch-size 8
```

## Batch UEA training
Run `train_uea.py` on multiple datasets in one command (extra args are passed through after `--`):

```bash
python scripts/run_uea_batch.py --datasets "SpokenArabicDigits FaceDetection Handwriting Heartbeat SelfRegulationSCP1 SelfRegulationSCP2" -- --device cpu --epochs 20
```

## FaceDetection quick run
`FaceDetection` now supports an auto profile in `train_uea.py` that applies tuned defaults (`n_fft=16`, `hop_length=4`, `val_split=0.3`, `ta_pair_mode=plain_cfc`) only when you did not override those flags.

```bash
python src/train_uea.py --dataset FaceDetection --dataset-profile auto --device cuda --pretrain-epochs 20 --epochs 60 --num-workers 8
```

## Heartbeat quick run
`Heartbeat` auto profile applies imbalance-aware defaults (`val_split=0.3`, `class_weight_mode=balanced`, `label_smoothing=0.0`) while keeping explicit CLI flags as highest priority. You can explicitly enable tri-view supervised finetune via `--supervised-views triview`.

```bash
python src/train_uea.py --dataset Heartbeat --dataset-profile auto --device cuda --pretrain-epochs 15 --epochs 80 --num-workers 0 --eval-num-workers 0
```

Tri-view with gated fusion:

```bash
python src/train_uea.py --dataset Heartbeat --dataset-profile none --supervised-views triview --triview-fusion gated --device cuda --epochs 80 --finetune-epochs 80 --num-workers 0 --eval-num-workers 0
```

For stronger imbalance mitigation, enable focal loss + logit adjustment:

```bash
python src/train_uea.py --dataset Heartbeat --dataset-profile auto --device cuda --loss-type focal --focal-gamma 2.0 --logit-adjustment train_prior --logit-adjust-tau 1.0 --logit-adjust-on-eval --epochs 100 --finetune-epochs 100 --num-workers 0 --eval-num-workers 0
```

## Heartbeat target-search
Use iterative search to progressively try stronger configs and stop early if target `test_acc` is reached:

```bash
python scripts/tune_heartbeat_target.py --device cuda --target-acc 0.90 --epochs 60 --finetune-epochs 60
```

## Handwriting quick run
`Handwriting` has very small training support (150 samples over 26 classes). Use `dataset-profile auto` (now defaults to `val_split=0.0`) to avoid over-splitting train data.

```bash
python src/train_uea.py --dataset Handwriting --dataset-profile auto --device cuda --supervised-views time --backbone all --res-blocks 2 --use-temporal-attn --use-shared-qk-attn --shared-qk-heads 4 --pretrain-epochs 0 --epochs 80 --finetune-epochs 80 --no-freeze-encoder --encoder-lr 1e-4 --head-lr 2e-4 --weight-decay 5e-4 --loss-type ce --label-smoothing 0.0 --val-split 0.0 --batch-size 64 --num-workers 0 --eval-num-workers 0
```

## HHAR real-world OOD preprocessing
Natural split (phone -> train, watch -> test):

```bash
python scripts/preprocess_hhar.py --sensor accelerometer --window-size 128 --stride 64
```

WOODS-style split (leave one device-model domain out as target, default 5s window at 100Hz):

```bash
python scripts/preprocess_hhar.py --protocol woods --sensor accelerometer --woods-target-model nexus4
```

WOODS-style 6-channel split (align `acc+gyro` windows and concatenate to `(N,6,T)`):

```bash
python scripts/preprocess_hhar.py --protocol woods --sensor both --woods-target-model nexus4
```

This writes:
- `dataset/all_datasets/HHAR/HHAR_TRAIN.pt`
- `dataset/all_datasets/HHAR/HHAR_TEST.pt`
- `dataset/all_datasets/HHAR/HHAR_meta.json`

Train/evaluate on HHAR:

```bash
python src/train_uea.py --dataset HHAR --dataset-profile auto --val-split-mode domain_stratified --device cuda --supervised-views triview --backbone all --use-temporal-attn --use-shared-qk-attn --epochs 40 --finetune-epochs 40 --pretrain-epochs 0 --batch-size 128 --num-workers 0 --eval-num-workers 0
```

For WOODS-style reporting, rerun preprocessing/training for each target model in `nexus4,s3,s3mini,lgwatch,gear` and average the held-out test metrics.

## Table-2 DG baselines (ERM / IRM / REx)
`train_uea.py` now supports standard DG objectives via:
- `--dg-method {erm,irm,rex}`
- `--dg-lambda`
- `--dg-min-group-size`
- `--dg-train-with-transforms` (optional; use only when dataset has no real domain ids)

One-command sweep for aligned baselines:

```bash
python scripts/run_dg_baselines_table2.py --dataset UWaveGestureLibrary --methods erm,irm,rex --seeds 42,43,44 --device cuda
```

This script writes per-seed and summary tables under `outputs_46/dg_baselines_table2/tables/`.

## Shared-QK ON/OFF ablation (same seeds + same efficiency protocol)
Run paired ON/OFF ablation with matched seeds and matched efficiency settings:

```bash
python scripts/run_shared_qk_ablation.py --dataset UWaveGestureLibrary --seeds 42,43,44 --device cuda --eff-device cuda --eff-batch-size 64 --eff-warmup 30 --eff-repeat 100
```

Outputs:
- `shared_qk_ablation_per_seed.csv`
- `shared_qk_ablation_summary.csv`
- `shared_qk_ablation_efficiency.csv`
- `shared_qk_ablation_protocol.json` (records the exact measurement protocol)

## STFT parameter sensitivity (local neighborhood)
This script runs the reviewer-facing sensitivity design around each dataset's default STFT setting:
- Handwriting: `n_fft in {16,32,64}`
- UWaveGestureLibrary: `n_fft in {128,256,512}`
- HHAR: `n_fft in {32,64,128}`

It enforces `win_length = n_fft` and `hop_length = n_fft / 4`, trains three variants (`baseline`, `triview`, `full`=TriView-TA), then runs `sweep_transforms.py` and exports:
- `stft_sensitivity_per_seed.csv`
- `stft_sensitivity_summary.csv`
- `stft_sensitivity_protocol.json`

Example (TFproject + CUDA):

```bash
python scripts/run_stft_sensitivity.py --device cuda --sweep-device cuda --datasets Handwriting,UWaveGestureLibrary,HHAR --seeds 42 --output-dir outputs_46/stft_sensitivity
```

If your main paper protocol needs extra train/sweep flags, append them once:

```bash
python scripts/run_stft_sensitivity.py --device cuda --sweep-device cuda --train-common-args "--use-temporal-attn --use-shared-qk-attn --shared-qk-heads 4" --sweep-common-args "--shift-fill border --severity-source train"
```

## Tests
```bash
pytest -q
```
