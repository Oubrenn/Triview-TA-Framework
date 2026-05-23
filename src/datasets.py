from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from .preprocessing import (
        PreprocessConfig,
        apply_per_channel as _apply_per_channel_core,
        build_augmented_triviews,
        build_triview_from_time,
        freq_view_from_mag,
        normalize_time_series,
        tf_view_from_mag,
    )
    from .transforms import (
        ColorParams,
        ScaleParams,
        ShiftParams,
        build_meta,
        make_coloring_gains,
    )
except ImportError:  # pragma: no cover - allow running as a script
    from preprocessing import (  # type: ignore
        PreprocessConfig,
        apply_per_channel as _apply_per_channel_core,
        build_augmented_triviews,
        build_triview_from_time,
        freq_view_from_mag,
        normalize_time_series,
        tf_view_from_mag,
    )
    from transforms import (  # type: ignore
        ColorParams,
        ScaleParams,
        ShiftParams,
        build_meta,
        make_coloring_gains,
    )


@dataclass(frozen=True)
class ViewConfig:
    n_fft: int = 256
    hop_length: int = 64
    win_length: Optional[int] = None
    window_name: str = "hann"
    center: bool = True
    magnitude_power: float = 1.0
    tf_log1p: bool = True
    tf_flatten: bool = True
    normalize_mode: str = "per_sample_channel"
    shift_mode: str = "zero"
    shift_bins: List[float] = None
    scale_ratios: List[float] = None
    color_bands: int = 8
    color_max_gain_db: float = 6.0
    color_active_bands: Optional[List[int]] = None
    color_max_gain_db_levels: Optional[List[float]] = None

    def __post_init__(self):
        object.__setattr__(self, "shift_bins", self.shift_bins or [3.0, -3.0])
        object.__setattr__(self, "scale_ratios", self.scale_ratios or [0.9, 1.1])
        if self.color_max_gain_db_levels is None:
            levels = [float(self.color_max_gain_db)]
        else:
            levels = []
            seen = set()
            for raw in self.color_max_gain_db_levels:
                value = float(raw)
                if value < 0:
                    raise ValueError("color_max_gain_db_levels must be non-negative.")
                key = round(value, 12)
                if key in seen:
                    continue
                seen.add(key)
                levels.append(value)
            if not levels:
                raise ValueError("color_max_gain_db_levels must contain at least one value.")
        object.__setattr__(self, "color_max_gain_db_levels", levels)
        object.__setattr__(self, "color_max_gain_db", max(levels))
        if self.color_bands <= 0:
            raise ValueError("color_bands must be positive.")
        if self.color_active_bands is not None:
            if len(self.color_active_bands) == 0:
                raise ValueError("color_active_bands must be non-empty when provided.")
            cleaned = sorted(set(int(idx) for idx in self.color_active_bands))
            for idx in cleaned:
                if idx < 0 or idx >= self.color_bands:
                    raise ValueError(
                        f"color_active_bands index {idx} out of range for color_bands={self.color_bands}."
                    )
            object.__setattr__(self, "color_active_bands", cleaned)
        if self.normalize_mode not in {"per_sample_channel", "none"}:
            raise ValueError(
                "normalize_mode must be one of {'per_sample_channel', 'none'}."
            )
        if self.window_name not in {"hann", "hamming"}:
            raise ValueError("window_name must be one of {'hann', 'hamming'}.")
        if self.magnitude_power <= 0:
            raise ValueError("magnitude_power must be positive.")
        if self.shift_mode not in {"zero", "circular", "border", "reflect"}:
            raise ValueError("shift_mode must be one of {'zero', 'circular', 'border', 'reflect'}.")

    def to_preprocess_config(self) -> PreprocessConfig:
        return PreprocessConfig(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_name=self.window_name,
            center=self.center,
            magnitude_power=self.magnitude_power,
            tf_log1p=self.tf_log1p,
            tf_flatten=self.tf_flatten,
            normalize_mode=self.normalize_mode,
        )


def _sample_transform_params(
    view_config: ViewConfig,
    num_bins: int,
    seed: Optional[int] = None,
    domain_id: Optional[int] = None,
) -> Dict[str, object]:
    sample_seed = int(seed) if seed is not None else int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    py_rng = random.Random(sample_seed)
    num_shift = len(view_config.shift_bins)
    num_scale = len(view_config.scale_ratios)
    num_color = len(view_config.color_max_gain_db_levels)
    total_domains = num_shift * num_scale * num_color
    if total_domains <= 0:
        raise ValueError("Domain grid must be non-empty.")

    if domain_id is None:
        shift_idx = py_rng.randrange(num_shift)
        scale_idx = py_rng.randrange(num_scale)
        color_idx = py_rng.randrange(num_color)
    else:
        domain_idx = int(domain_id)
        if domain_idx < 0 or domain_idx >= total_domains:
            raise ValueError(f"domain_id out of range: {domain_idx} not in [0, {total_domains - 1}]")
        shift_idx = domain_idx // (num_scale * num_color)
        rem = domain_idx % (num_scale * num_color)
        scale_idx = rem // num_color
        color_idx = rem % num_color

    shift_bins = float(view_config.shift_bins[shift_idx])
    ratio = float(view_config.scale_ratios[scale_idx])
    color_max_db = float(view_config.color_max_gain_db_levels[color_idx])
    color_generator = torch.Generator().manual_seed(sample_seed)
    gains, band_gains = make_coloring_gains(
        num_bins=num_bins,
        bands=view_config.color_bands,
        max_gain_db=color_max_db,
        return_band_gains=True,
        active_bands=view_config.color_active_bands,
        generator=color_generator,
    )
    resolved_domain_id = shift_idx * (num_scale * num_color) + scale_idx * num_color + color_idx
    return {
        "seed": sample_seed,
        "shift_bins": shift_bins,
        "shift_severity_id": shift_idx,
        "scale_ratio": ratio,
        "scale_severity_id": scale_idx,
        "color_gains": gains,
        "color_band_gains": band_gains,
        "color_max_gain_db": color_max_db,
        "color_severity_id": color_idx,
        "domain_id": resolved_domain_id,
        "domain_tag": f"sh{shift_idx}-sc{scale_idx}-co{color_idx}",
    }


class SyntheticTimeSeriesDataset(Dataset):
    """Simple dataset for MVP validation with controllable transforms."""

    def __init__(
        self,
        num_samples: int,
        length: int,
        view_config: Optional[ViewConfig] = None,
    ) -> None:
        self.num_samples = num_samples
        self.length = length
        self.view_config = view_config or ViewConfig()
        self.preprocess_config = self.view_config.to_preprocess_config()
        self.base = torch.randn(num_samples, length)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = normalize_time_series(self.base[idx], mode=self.view_config.normalize_mode)
        draw = _sample_transform_params(
            self.view_config,
            num_bins=x.shape[-1] // 2 + 1,
        )
        views = build_augmented_triviews(
            x=x,
            config=self.preprocess_config,
            shift_bins=float(draw["shift_bins"]),
            scale_ratio=float(draw["scale_ratio"]),
            color_gains=draw["color_gains"],
            shift_mode=self.view_config.shift_mode,
        )

        color_meta = ColorParams(gains=draw["color_band_gains"])
        meta = {
            "seed": torch.tensor(draw["seed"], dtype=torch.long),
            "shift_mode": self.view_config.shift_mode,
            "shift": {
                **build_meta(ScaleParams(1.0), ShiftParams(float(draw["shift_bins"])), color_meta),
                "severity_id": torch.tensor(int(draw["shift_severity_id"]), dtype=torch.long),
            },
            "scale": {
                **build_meta(ScaleParams(float(draw["scale_ratio"])), ShiftParams(0), color_meta),
                "severity_id": torch.tensor(int(draw["scale_severity_id"]), dtype=torch.long),
            },
            "color": build_meta(ScaleParams(1.0), ShiftParams(0), color_meta),
            "domain": {
                "id": torch.tensor(int(draw["domain_id"]), dtype=torch.long),
                "shift_id": torch.tensor(int(draw["shift_severity_id"]), dtype=torch.long),
                "scale_id": torch.tensor(int(draw["scale_severity_id"]), dtype=torch.long),
                "color_id": torch.tensor(int(draw["color_severity_id"]), dtype=torch.long),
                "tag": draw["domain_tag"],
            },
            "transform_params": {
                "b": torch.tensor(float(draw["shift_bins"]), dtype=torch.float32),
                "rho": torch.tensor(float(draw["scale_ratio"]), dtype=torch.float32),
                "g_db": torch.tensor(float(draw["color_max_gain_db"]), dtype=torch.float32),
                "color_id": torch.tensor(int(draw["color_severity_id"]), dtype=torch.long),
            },
        }
        meta["color"]["severity_id"] = torch.tensor(int(draw["color_severity_id"]), dtype=torch.long)
        meta["color"]["max_gain_db"] = torch.tensor(float(draw["color_max_gain_db"]), dtype=torch.float32)

        return {
            "x_time": x,
            **views,
            "meta": meta,
        }


UEA_DATASETS = {
    "UWaveGestureLibrary",
    "SpokenArabicDigits",
    "JapaneseVowels",
    "FaceDetection",
    "Handwriting",
    "Heartbeat",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
}
HHAR_DATASETS = {"HHAR"}
SUPPORTED_DATASETS = UEA_DATASETS | HHAR_DATASETS


def _parse_uea_values(segment: str) -> List[float]:
    values: List[float] = []
    for item in segment.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "?":
            values.append(float("nan"))
        else:
            values.append(float(item))
    return values


def _parse_uea_ts_file(path: Path) -> Tuple[List[List[List[float]]], List[str], Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"UEA .ts file not found: {path}")

    meta: Dict[str, object] = {
        "problem_name": None,
        "timestamps": False,
        "missing": False,
        "univariate": None,
        "dimensions": None,
        "equal_length": None,
        "series_length": None,
        "class_labels": None,
    }
    data: List[List[List[float]]] = []
    labels: List[str] = []
    in_data = False

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            lower = line.lower()
            if not in_data:
                if lower.startswith("@problemname"):
                    meta["problem_name"] = line.split(maxsplit=1)[1] if len(line.split()) > 1 else None
                elif lower.startswith("@timestamps"):
                    meta["timestamps"] = line.split()[1].lower() == "true"
                elif lower.startswith("@missing"):
                    meta["missing"] = line.split()[1].lower() == "true"
                elif lower.startswith("@univariate"):
                    meta["univariate"] = line.split()[1].lower() == "true"
                elif lower.startswith("@dimensions"):
                    meta["dimensions"] = int(line.split()[1])
                elif lower.startswith("@equallength"):
                    meta["equal_length"] = line.split()[1].lower() == "true"
                elif lower.startswith("@serieslength"):
                    meta["series_length"] = int(line.split()[1])
                elif lower.startswith("@classlabel"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].lower() == "true":
                        meta["class_labels"] = parts[2:] or None
                elif lower.startswith("@data"):
                    in_data = True
                continue

            if meta["timestamps"]:
                raise ValueError("Timestamped .ts files are not supported yet.")

            parts = line.split(":")
            label = None
            if meta["class_labels"] is not None:
                label = parts[-1].strip()
                parts = parts[:-1]

            series = [_parse_uea_values(part) for part in parts]
            if meta["dimensions"] is not None and len(series) != meta["dimensions"]:
                raise ValueError(
                    f"Expected {meta['dimensions']} dimensions but got {len(series)} in {path.name}."
                )

            data.append(series)
            if label is not None:
                labels.append(label)

    return data, labels, meta


_UEA_CACHE_VERSION = 1


def _uea_cache_path(ts_path: Path, pad_to_max: bool) -> Path:
    suffix = "pad1" if pad_to_max else "pad0"
    return ts_path.parent / f"{ts_path.stem}_{suffix}.cache.pt"


def _source_signature(path: Path) -> Tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def _load_cached_uea_split(ts_path: Path, pad_to_max: bool) -> Optional[Dict[str, object]]:
    cache_path = _uea_cache_path(ts_path, pad_to_max)
    if not cache_path.exists():
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version", -1)) != _UEA_CACHE_VERSION:
        return None
    if payload.get("source_signature") != _source_signature(ts_path):
        return None
    required_keys = {"meta", "class_labels", "labels", "lengths", "data"}
    if not required_keys.issubset(set(payload.keys())):
        return None
    return payload


def _save_cached_uea_split(
    ts_path: Path,
    pad_to_max: bool,
    *,
    meta: Dict[str, object],
    class_labels: List[str],
    labels: torch.Tensor,
    lengths: torch.Tensor,
    data: List[torch.Tensor],
) -> None:
    cache_path = _uea_cache_path(ts_path, pad_to_max)
    payload = {
        "version": _UEA_CACHE_VERSION,
        "source_signature": _source_signature(ts_path),
        "meta": meta,
        "class_labels": class_labels,
        "labels": labels.cpu(),
        "lengths": lengths.cpu(),
        "data": [tensor.cpu() for tensor in data],
    }
    try:
        torch.save(payload, cache_path)
    except Exception:
        # Cache writes should never block training.
        pass


class UEATimeSeriesDataset(Dataset):
    def __init__(
        self,
        name: str,
        split: str = "train",
        root_dir: Optional[Path] = None,
        normalize: bool = True,
        pad_to_max: bool = True,
        return_freq: bool = False,
        view_config: Optional[ViewConfig] = None,
        use_cache: bool = True,
    ) -> None:
        if name not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {name}")
        split_key = split.lower()
        if split_key not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        base_dir = root_dir or Path(__file__).resolve().parents[1] / "dataset" / "all_datasets"
        self.domain_ids = None
        if name in HHAR_DATASETS:
            hhar_path = base_dir / name / f"{name}_{split_key.upper()}.pt"
            if not hhar_path.exists():
                raise FileNotFoundError(
                    f"HHAR split file not found: {hhar_path}. "
                    "Run scripts/preprocess_hhar.py first."
                )
            payload = torch.load(hhar_path, map_location="cpu")
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid HHAR split payload: {hhar_path}")
            required = {"data", "labels", "lengths", "class_labels"}
            if not required.issubset(set(payload.keys())):
                raise ValueError(
                    f"HHAR split file missing keys {required - set(payload.keys())}: {hhar_path}"
                )
            data_tensor = payload["data"]
            if not torch.is_tensor(data_tensor) or data_tensor.dim() != 3:
                raise ValueError("HHAR 'data' must be a tensor with shape (N, C, T).")
            self.data = [data_tensor[i].to(dtype=torch.float32).cpu() for i in range(data_tensor.shape[0])]
            self.labels = payload["labels"].to(dtype=torch.long).cpu()
            self.lengths = payload["lengths"].to(dtype=torch.long).cpu()
            self.class_labels = [str(label) for label in payload["class_labels"]]
            self.label_to_index = {label: idx for idx, label in enumerate(self.class_labels)}
            self.meta = dict(payload.get("meta", {}))
            domain_ids = payload.get("domain_ids")
            if domain_ids is not None:
                if not torch.is_tensor(domain_ids):
                    raise ValueError("HHAR 'domain_ids' must be a tensor when provided.")
                self.domain_ids = domain_ids.to(dtype=torch.long).cpu()
        else:
            ts_path = base_dir / name / f"{name}_{split_key.upper()}.ts"

            cached = _load_cached_uea_split(ts_path, pad_to_max) if use_cache else None
            if cached is not None:
                meta = cached["meta"]
                class_labels = cached["class_labels"]
                labels = cached["labels"]
                lengths = cached["lengths"]
                data_tensors = cached["data"]

                self.meta = dict(meta) if isinstance(meta, dict) else {}
                self.class_labels = [str(label) for label in class_labels]
                self.label_to_index = {label: idx for idx, label in enumerate(self.class_labels)}
                self.labels = labels.to(dtype=torch.long).cpu()
                self.lengths = lengths.to(dtype=torch.long).cpu()
                self.data = [tensor.to(dtype=torch.float32).cpu() for tensor in data_tensors]
            else:
                series_data, label_strings, meta = _parse_uea_ts_file(ts_path)
                self.meta = meta
                self.class_labels = meta.get("class_labels") or sorted(set(label_strings))
                self.label_to_index = {label: idx for idx, label in enumerate(self.class_labels)}
                self.labels = torch.tensor([self.label_to_index[label] for label in label_strings], dtype=torch.long)

                lengths_list = [len(sample[0]) if sample else 0 for sample in series_data]
                max_len = max(lengths_list) if pad_to_max and lengths_list else None
                data_tensors: List[torch.Tensor] = []
                for sample, length in zip(series_data, lengths_list):
                    x = torch.tensor(sample, dtype=torch.float32)
                    if max_len is not None and x.shape[-1] < max_len:
                        x = F.pad(x, (0, max_len - x.shape[-1]))
                    data_tensors.append(x)

                self.data = data_tensors
                self.lengths = torch.tensor(lengths_list, dtype=torch.long)
                if use_cache:
                    _save_cached_uea_split(
                        ts_path,
                        pad_to_max,
                        meta=self.meta,
                        class_labels=self.class_labels,
                        labels=self.labels,
                        lengths=self.lengths,
                        data=self.data,
                    )
        self.normalize = normalize
        self.return_freq = return_freq
        self.view_config = view_config or ViewConfig()
        self.preprocess_config = self.view_config.to_preprocess_config()

    def __len__(self) -> int:
        return len(self.data)

    def _normalize(self, x: torch.Tensor, length: int) -> torch.Tensor:
        return normalize_time_series(x, length=length, mode=self.view_config.normalize_mode)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.data[idx]
        length = int(self.lengths[idx].item())
        if self.normalize:
            x = self._normalize(x, length)

        item: Dict[str, torch.Tensor] = {
            "x_time": x,
            "y": self.labels[idx],
            "length": self.lengths[idx],
        }
        if self.domain_ids is not None:
            item["meta"] = {"domain_id": self.domain_ids[idx]}

        if self.return_freq:
            views = build_triview_from_time(x, self.preprocess_config)
            item["x_freq"] = views["x_freq"]
            item["x_tf"] = views["x_tf"]

        return item


def _apply_per_channel(x: torch.Tensor, fn, *args, **kwargs) -> torch.Tensor:
    return _apply_per_channel_core(x, fn, *args, **kwargs)


def _freq_view_from_mag(stft: torch.Tensor) -> torch.Tensor:
    return freq_view_from_mag(stft)


def _tf_view_from_mag(stft: torch.Tensor) -> torch.Tensor:
    return tf_view_from_mag(stft, flatten=True)


def _stft_view(x: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    config = PreprocessConfig(n_fft=n_fft, hop_length=hop_length)
    views = build_triview_from_time(x, config)
    return views["x_freq"]


def _stft_tf_view(x: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    config = PreprocessConfig(n_fft=n_fft, hop_length=hop_length)
    views = build_triview_from_time(x, config)
    return views["x_tf"]


class UEAPretrainDataset(Dataset):
    def __init__(
        self,
        name: str,
        split: str = "train",
        root_dir: Optional[Path] = None,
        pad_to_max: bool = True,
        view_config: Optional[ViewConfig] = None,
        base_seed: Optional[int] = None,
        source_domain_ids: Optional[List[int]] = None,
        use_cache: bool = True,
    ) -> None:
        self.view_config = view_config or ViewConfig()
        self.total_domains = (
            len(self.view_config.shift_bins)
            * len(self.view_config.scale_ratios)
            * len(self.view_config.color_max_gain_db_levels)
        )
        if self.total_domains <= 0:
            raise ValueError("Domain grid must be non-empty for pretraining.")
        if source_domain_ids:
            cleaned = sorted(set(int(v) for v in source_domain_ids))
            for domain_id in cleaned:
                if domain_id < 0 or domain_id >= self.total_domains:
                    raise ValueError(
                        f"source_domain_ids contains out-of-range id {domain_id}; "
                        f"valid range is [0, {self.total_domains - 1}]"
                    )
            self.source_domain_ids = cleaned
        else:
            self.source_domain_ids = None
        self.base = UEATimeSeriesDataset(
            name=name,
            split=split,
            root_dir=root_dir,
            normalize=True,
            pad_to_max=pad_to_max,
            return_freq=False,
            view_config=self.view_config,
            use_cache=use_cache,
        )
        # Expose label/length metadata for stratified splitting and weighted sampling.
        self.labels = self.base.labels
        self.lengths = self.base.lengths
        self.class_labels = self.base.class_labels
        self.preprocess_config = self.view_config.to_preprocess_config()
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base[idx]
        x = item["x_time"]
        fixed_seed = None if self.base_seed is None else int(self.base_seed + idx)
        selected_domain_id = None
        if self.source_domain_ids is not None:
            if fixed_seed is None:
                draw_idx = int(torch.randint(0, len(self.source_domain_ids), (1,), dtype=torch.int64).item())
            else:
                domain_rng = random.Random(fixed_seed + 7919)
                draw_idx = domain_rng.randrange(len(self.source_domain_ids))
            selected_domain_id = int(self.source_domain_ids[draw_idx])
        draw = _sample_transform_params(
            self.view_config,
            num_bins=x.shape[-1] // 2 + 1,
            seed=fixed_seed,
            domain_id=selected_domain_id,
        )
        views = build_augmented_triviews(
            x=x,
            config=self.preprocess_config,
            shift_bins=float(draw["shift_bins"]),
            scale_ratio=float(draw["scale_ratio"]),
            color_gains=draw["color_gains"],
            shift_mode=self.view_config.shift_mode,
        )

        color_meta = ColorParams(gains=draw["color_band_gains"])
        meta = {
            "seed": torch.tensor(draw["seed"], dtype=torch.long),
            "shift_mode": self.view_config.shift_mode,
            "shift": {
                **build_meta(ScaleParams(1.0), ShiftParams(float(draw["shift_bins"])), color_meta),
                "severity_id": torch.tensor(int(draw["shift_severity_id"]), dtype=torch.long),
            },
            "scale": {
                **build_meta(ScaleParams(float(draw["scale_ratio"])), ShiftParams(0), color_meta),
                "severity_id": torch.tensor(int(draw["scale_severity_id"]), dtype=torch.long),
            },
            "color": build_meta(ScaleParams(1.0), ShiftParams(0), color_meta),
            "domain": {
                "id": torch.tensor(int(draw["domain_id"]), dtype=torch.long),
                "shift_id": torch.tensor(int(draw["shift_severity_id"]), dtype=torch.long),
                "scale_id": torch.tensor(int(draw["scale_severity_id"]), dtype=torch.long),
                "color_id": torch.tensor(int(draw["color_severity_id"]), dtype=torch.long),
                "tag": draw["domain_tag"],
            },
            "transform_params": {
                "b": torch.tensor(float(draw["shift_bins"]), dtype=torch.float32),
                "rho": torch.tensor(float(draw["scale_ratio"]), dtype=torch.float32),
                "g_db": torch.tensor(float(draw["color_max_gain_db"]), dtype=torch.float32),
                "color_id": torch.tensor(int(draw["color_severity_id"]), dtype=torch.long),
            },
        }
        meta["color"]["severity_id"] = torch.tensor(int(draw["color_severity_id"]), dtype=torch.long)
        meta["color"]["max_gain_db"] = torch.tensor(float(draw["color_max_gain_db"]), dtype=torch.float32)

        return {
            "x_time": x,
            "y": item["y"],
            "length": item["length"],
            **views,
            "meta": meta,
        }
