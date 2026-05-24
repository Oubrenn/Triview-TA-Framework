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

UWaveGestureLibrary:

```bash
python src/train_uea.py --dataset UWaveGestureLibrary --device cuda --seed 42 --dataset-profile auto --supervised-views triview --backbone all --triview-fusion gated --use-temporal-attn --use-shared-qk-attn --shared-qk-heads 4 --pretrain-epochs 0 --epochs 20 --finetune-epochs 20 --batch-size 32 --num-workers 0 --eval-num-workers 0
```

SpokenArabicDigits:

```bash
python src/train_uea.py --dataset SpokenArabicDigits --device cuda --seed 42 --dataset-profile auto --supervised-views triview --backbone all --triview-fusion gated --use-temporal-attn --use-shared-qk-attn --shared-qk-heads 4 --pretrain-epochs 0 --epochs 20 --finetune-epochs 20 --batch-size 32 --num-workers 0 --eval-num-workers 0
```

FaceDetection:

```bash
python src/train_uea.py --dataset FaceDetection --device cuda --seed 42 --dataset-profile auto --supervised-views triview --backbone all --triview-fusion gated --use-temporal-attn --use-shared-qk-attn --shared-qk-heads 4 --pretrain-epochs 0 --epochs 20 --finetune-epochs 20 --batch-size 32 --num-workers 0 --eval-num-workers 0
```

Handwriting:

```bash
python src/train_uea.py --dataset Handwriting --device cuda --seed 42 --dataset-profile auto --supervised-views triview --backbone all --triview-fusion gated --use-temporal-attn --use-shared-qk-attn --shared-qk-heads 4 --pretrain-epochs 0 --epochs 20 --finetune-epochs 20 --batch-size 32 --num-workers 0 --eval-num-workers 0
```

Heartbeat:

```bash
python src/train_uea.py --dataset Heartbeat --device cuda --seed 42 --dataset-profile auto --supervised-views triview --backbone all --triview-fusion gated --use-temporal-attn --use-shared-qk-attn --shared-qk-heads 4 --pretrain-epochs 0 --epochs 20 --finetune-epochs 20 --batch-size 32 --num-workers 0 --eval-num-workers 0
```

SelfRegulationSCP1:

```bash
python src/train_uea.py --dataset SelfRegulationSCP1 --device cuda --seed 42 --dataset-profile auto --supervised-views triview --backbone all --triview-fusion gated --use-temporal-attn --use-shared-qk-attn --shared-qk-heads 4 --pretrain-epochs 0 --epochs 20 --finetune-epochs 20 --batch-size 32 --num-workers 0 --eval-num-workers 0
```

SelfRegulationSCP2:

```bash
python src/train_uea.py --dataset SelfRegulationSCP2 --device cuda --seed 42 --dataset-profile auto --supervised-views triview --backbone all --triview-fusion gated --use-temporal-attn --use-shared-qk-attn --shared-qk-heads 4 --pretrain-epochs 0 --epochs 20 --finetune-epochs 20 --batch-size 32 --num-workers 0 --eval-num-workers 0
```

`--dataset-profile auto` only applies small dataset-specific defaults such as STFT size
and validation split.

## Tests

```bash
pip install pytest
pytest -q
```
