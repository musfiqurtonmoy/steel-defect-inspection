"""Load and hold run configuration from a YAML file."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    db_url: str = "sqlite:///data/manifest.db"
    image_size: int = 224
    min_side: int = 32
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42


@dataclass
class TrainConfig:
    arch: str = "resnet18"  # resnet18 | mobilenet_v3_small
    pretrained: bool = True
    epochs: int = 15
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 4
    num_workers: int = 2
    seed: int = 42
    out_dir: str = "artifacts"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None) -> Config:
    if path is None:
        return Config()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config(
        data=DataConfig(**raw.get("data", {})),
        train=TrainConfig(**raw.get("train", {})),
    )
