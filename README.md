# TriView-TA Framework

Source code for the TriView-TA time-series classification framework.

## What is included

- `src/`: dataset loaders, transforms, models, losses, and training entry points.
- `scripts/` and `tools/`: batch runs, evaluation helpers, and plotting utilities.
- `tests/`: lightweight checks for dataset and transform logic.

Data, checkpoints, generated figures, and experiment outputs are intentionally not tracked.

## Install

```bash
pip install -r requirements.txt
```

## Reproduce the 7 UEA Datasets

The seven datasets are `UWaveGestureLibrary`, `SpokenArabicDigits`, `FaceDetection`,
`Handwriting`, `Heartbeat`, `SelfRegulationSCP1`, and `SelfRegulationSCP2`.

```bash
python scripts/run_uea_batch.py --datasets "UWaveGestureLibrary SpokenArabicDigits FaceDetection Handwriting Heartbeat SelfRegulationSCP1 SelfRegulationSCP2" -- --device cuda --seed 42 --dataset-profile auto --supervised-views triview --backbone all --triview-fusion gated --use-temporal-attn --use-shared-qk-attn --shared-qk-heads 4 --pretrain-epochs 0 --epochs 20 --finetune-epochs 20 --batch-size 32 --num-workers 0 --eval-num-workers 0
```

`--dataset-profile auto` only applies small dataset-specific defaults such as STFT size
and validation split.

## Tests

```bash
pytest -q
```
