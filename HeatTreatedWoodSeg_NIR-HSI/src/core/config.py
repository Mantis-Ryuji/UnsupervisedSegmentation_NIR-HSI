from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml


# =========================================================
# Dataclass 定義
#   - config.yaml の構造を型付きで扱うための薄いラッパ
#   - 既存の config.yaml に無い項目はデフォルト値で埋める
# =========================================================

@dataclass
class ModelConfig:
    """
    model セクション用の設定コンテナ。

    config.yaml の `model:` を型付きで表現。
    """
    # -------------------------
    # Encoder (Transformer)
    # -------------------------
    seq_len: int = 256
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # -------------------------
    # Latent / Masking
    # -------------------------
    latent_dim: int = 16
    n_patches: int = 32
    n_mask: int = 24


@dataclass
class TrainingConfig:
    """
    training セクション用の設定コンテナ。
    """
    seed: int = 42

    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    base_lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

    warmup_epochs: int = 1
    min_lr_scale: float = 0.05

    epochs: int = 150
    early_stop_patience: int = 15


@dataclass
class ClusteringConfig:
    """
    clustering セクション用の設定コンテナ。
    """
    k_max: int = 50
    chunk: int = 5_000_000


@dataclass
class TestConfig:
    """
    test セクション用の設定コンテナ。
    """
    weights_path: Optional[str] = None

@dataclass
class ProjectConfig:
    """
    リポジトリ全体の設定ビュー。

    - raw: YAML をそのまま保持した dict
    - model, training, clustering, test, report: 各セクションの型付き view
    """
    raw: Dict[str, Any]
    model: ModelConfig
    training: TrainingConfig
    clustering: ClusteringConfig
    test: TestConfig

    def get_section(self, name: str, default: Any = None) -> Any:
        """
        任意のセクションを生の dict として取り出したい場合のヘルパ。
        （まだ dataclass 化していないセクションにアクセスするとき用）
        """
        return self.raw.get(name, default)


# =========================================================
# 内部ヘルパ（dict -> dataclass 変換）
# =========================================================

def _as_model_cfg(d: Mapping[str, Any]) -> ModelConfig:
    """
    model dict -> ModelConfig 変換。
    """
    base = ModelConfig()
    return ModelConfig(
        seq_len=int(d.get("seq_len", base.seq_len)),
        d_model=int(d.get("d_model", base.d_model)),
        nhead=int(d.get("nhead", base.nhead)),
        num_layers=int(d.get("num_layers", base.num_layers)),
        dim_feedforward=int(d.get("dim_feedforward", base.dim_feedforward)),
        dropout=float(d.get("dropout", base.dropout)),
        latent_dim=int(d.get("latent_dim", base.latent_dim)),
        n_patches=int(d.get("n_patches", base.n_patches)),
        n_mask=int(d.get("n_mask", base.n_mask)),
    )


def _as_training_cfg(d: Mapping[str, Any]) -> TrainingConfig:
    base = TrainingConfig()
    betas_raw = d.get("betas", (base.betas[0], base.betas[1]))
    if isinstance(betas_raw, (list, tuple)) and len(betas_raw) >= 2:
        betas = (float(betas_raw[0]), float(betas_raw[1]))
    else:
        betas = base.betas

    return TrainingConfig(
        seed=int(d.get("seed", base.seed)),
        batch_size=int(d.get("batch_size", base.batch_size)),
        num_workers=int(d.get("num_workers", base.num_workers)),
        pin_memory=bool(d.get("pin_memory", base.pin_memory)),
        persistent_workers=bool(d.get("persistent_workers", base.persistent_workers)),
        base_lr=float(d.get("base_lr", base.base_lr)),
        weight_decay=float(d.get("weight_decay", base.weight_decay)),
        betas=betas,
        eps=float(d.get("eps", base.eps)),
        warmup_epochs=int(d.get("warmup_epochs", base.warmup_epochs)),
        min_lr_scale=float(d.get("min_lr_scale", base.min_lr_scale)),
        epochs=int(d.get("epochs", base.epochs)),
        early_stop_patience=int(d.get("early_stop_patience", base.early_stop_patience)),
    )


def _as_clustering_cfg(d: Mapping[str, Any]) -> ClusteringConfig:
    base = ClusteringConfig()
    return ClusteringConfig(
        k_max=int(d.get("k_max", base.k_max)),
        chunk=int(d.get("chunk", base.chunk)),
    )


def _as_test_cfg(d: Mapping[str, Any]) -> TestConfig:
    base = TestConfig()
    weights = d.get("weights_path", base.weights_path)
    if weights is not None:
        weights = str(weights)
    return TestConfig(weights_path=weights)


# =========================================================
# 公開 API：config.yaml の読み込み
# =========================================================

_DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """
    config.yaml を読み込んで ProjectConfig を返す。
    """
    if path is None:
        path = _DEFAULT_CONFIG_PATH
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"config.yaml が見つかりません: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    model_dict      = data.get("model", {}) or {}
    training_dict   = data.get("training", {}) or {}
    clustering_dict = data.get("clustering", {}) or {}
    test_dict       = data.get("test", {}) or {}

    model_cfg      = _as_model_cfg(model_dict)
    training_cfg   = _as_training_cfg(training_dict)
    clustering_cfg = _as_clustering_cfg(clustering_dict)
    test_cfg       = _as_test_cfg(test_dict)

    return ProjectConfig(
        raw=data,
        model=model_cfg,
        training=training_cfg,
        clustering=clustering_cfg,
        test=test_cfg,
    )