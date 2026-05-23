import argparse
import csv
import sys
from pathlib import Path

import torch
from thop import profile

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import UEATimeSeriesDataset, ViewConfig  # noqa: E402
from train_uea import UEAClassifier, UEATriViewClassifier  # noqa: E402


DATASET_STFT = {
    "UWaveGestureLibrary": (256, 64),
    "JapaneseVowels": (16, 4),
    "SpokenArabicDigits": (16, 4),
    "Handwriting": (32, 8),
    "FaceDetection": (16, 4),
    "Heartbeat": (256, 64),
}

LATENCY_MS = {
    ("UWaveGestureLibrary", "InceptionTime"): 1.460,
    ("UWaveGestureLibrary", "All + Attn (time-only)"): 15.846,
    ("UWaveGestureLibrary", "TriView-All + Attn"): 27.843,
    ("JapaneseVowels", "InceptionTime"): 1.839,
    ("JapaneseVowels", "All + Attn (time-only)"): 8.152,
    ("JapaneseVowels", "TriView-All + Attn"): 27.920,
    ("SpokenArabicDigits", "InceptionTime"): 1.366,
    ("SpokenArabicDigits", "All + Attn (time-only)"): 7.036,
    ("SpokenArabicDigits", "TriView-All + Attn"): 27.956,
    ("Handwriting", "InceptionTime"): 1.743,
    ("Handwriting", "All + Attn (time-only)"): 10.115,
    ("Handwriting", "TriView-All + Attn"): 24.392,
    ("FaceDetection", "InceptionTime"): 1.610,
    ("FaceDetection", "All + Attn (time-only)"): 8.556,
    ("FaceDetection", "TriView-All + Attn"): 24.035,
    ("Heartbeat", "InceptionTime"): 1.884,
    ("Heartbeat", "All + Attn (time-only)"): 24.175,
    ("Heartbeat", "TriView-All + Attn"): 34.102,
}

MODEL_ORDER = (
    "InceptionTime",
    "All + Attn (time-only)",
    "TriView-All + Attn",
)


def count_params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def build_models(
    input_dim_time: int,
    input_dim_freq: int,
    input_dim_tf: int,
    num_classes: int,
):
    common = dict(
        hidden_dim=64,
        embed_dim=128,
        num_heads=4,
        res_blocks=2,
        fuse_dropout=0.1,
        head_dropout=0.1,
    )

    inception = UEAClassifier(
        input_dim=input_dim_time,
        num_classes=num_classes,
        backbone="inception",
        use_temporal_attn=False,
        use_shared_qk_attn=False,
        **common,
    ).eval()

    all_attn = UEAClassifier(
        input_dim=input_dim_time,
        num_classes=num_classes,
        backbone="all",
        use_temporal_attn=True,
        use_shared_qk_attn=True,
        **common,
    ).eval()

    tri_all_attn = UEATriViewClassifier(
        input_dim_time=input_dim_time,
        input_dim_freq=input_dim_freq,
        input_dim_tf=input_dim_tf,
        num_classes=num_classes,
        backbone="all",
        use_temporal_attn=True,
        use_shared_qk_attn=True,
        triview_fusion="gated",
        gate_hidden_dim=64,
        gate_dropout=0.0,
        gate_temperature=1.0,
        **common,
    ).eval()

    return {
        "InceptionTime": inception,
        "All + Attn (time-only)": all_attn,
        "TriView-All + Attn": tri_all_attn,
    }


def format_md(rows):
    lines = []
    lines.append("| Dataset | Model | Params (M) | MACs (G) | Batch latency (ms) |")
    lines.append("|---|---|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['Dataset']} | {row['Model']} | {row['Params (M)']:.3f} | "
            f"{row['MACs (G)']:.3f} | {row['Batch latency (ms)']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "outputs_46" / "tables" / "complexity_six_with_macs.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "outputs_46" / "tables" / "complexity_six_with_macs.md",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split used to derive representative input shapes.",
    )
    args = parser.parse_args()

    rows = []
    for dataset, (n_fft, hop_length) in DATASET_STFT.items():
        view_cfg = ViewConfig(
            n_fft=n_fft,
            hop_length=hop_length,
            tf_flatten=True,
            tf_log1p=True,
        )
        ds = UEATimeSeriesDataset(
            name=dataset,
            split=args.split,
            pad_to_max=True,
            return_freq=True,
            view_config=view_cfg,
        )
        sample = ds[0]
        x_time = sample["x_time"].unsqueeze(0)
        x_freq = sample["x_freq"].unsqueeze(0)
        x_tf = sample["x_tf"].unsqueeze(0)

        models = build_models(
            input_dim_time=int(x_time.shape[1]),
            input_dim_freq=int(x_freq.shape[1]),
            input_dim_tf=int(x_tf.shape[1]),
            num_classes=len(ds.class_labels),
        )

        inputs = {
            "InceptionTime": (x_time,),
            "All + Attn (time-only)": (x_time,),
            "TriView-All + Attn": (x_time, x_freq, x_tf),
        }

        for model_name in MODEL_ORDER:
            model = models[model_name]
            macs, _ = profile(model, inputs=inputs[model_name], verbose=False)
            rows.append(
                {
                    "Dataset": dataset,
                    "Model": model_name,
                    "Params (M)": round(count_params_m(model), 3),
                    "MACs (G)": round(float(macs) / 1e9, 3),
                    "Batch latency (ms)": float(LATENCY_MS[(dataset, model_name)]),
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                    "x_time_shape": str(tuple(x_time.shape)),
                    "x_freq_shape": str(tuple(x_freq.shape)),
                    "x_tf_shape": str(tuple(x_tf.shape)),
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Dataset",
                "Model",
                "Params (M)",
                "MACs (G)",
                "Batch latency (ms)",
                "n_fft",
                "hop_length",
                "x_time_shape",
                "x_freq_shape",
                "x_tf_shape",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(format_md(rows), encoding="utf-8")

    print(f"saved_csv={args.output_csv}")
    print(f"saved_md={args.output_md}")


if __name__ == "__main__":
    main()
