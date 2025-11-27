from __future__ import annotations

from pathlib import Path
from typing import Final, Literal


# =========================================================
# リポジトリルート
# =========================================================
ROOT_DIR: Final[Path] = Path(__file__).resolve().parents[2]


# =========================================================
# 基本ディレクトリ
# =========================================================
DATA_DIR:    Final[Path] = ROOT_DIR / "data"
RUNS_DIR:    Final[Path] = ROOT_DIR / "runs"
IMAGES_DIR:  Final[Path] = ROOT_DIR / "images"
CONFIG_PATH: Final[Path] = ROOT_DIR / "config.yaml"

# data 下の確定ディレクトリ
WHITE_DARK_DIR: Final[Path] = DATA_DIR / "white_dark"
SPECTRA_DIR:    Final[Path] = DATA_DIR / "spectra"
LATENT_DIR:     Final[Path] = DATA_DIR / "latent"

TRAIN_DIR: Final[Path] = DATA_DIR / "train"
VAL_DIR:   Final[Path] = DATA_DIR / "val"
TEST_DIR:  Final[Path] = DATA_DIR / "test"


# =========================================================
# split 系ユーティリティ
# =========================================================
def get_split_dir(split: Literal["train", "val", "test"]) -> Path:
    return DATA_DIR / split


def get_sample_dir(split: str) -> Path:
    return get_split_dir(split) / "samples"


def get_mask_dir(split: str) -> Path:
    return get_split_dir(split) / "masks"


def get_label_dir(split: str) -> Path:
    return get_split_dir(split) / "labels"


# =========================================================
# データセット I/O パス
# =========================================================
def get_reflectance_path(
    split: str,
    *,
    snv: bool,
    downsampled: bool = False,
) -> Path:
    """reflectance[_snv][_downsampled].npy の path を返す"""
    base = get_split_dir(split)
    if snv and downsampled:
        return base / "reflectance_snv_downsampled.npy"
    if snv:
        return base / "reflectance_snv.npy"
    return base / "reflectance.npy"


def get_latent_path(split: str) -> Path:
    """latent Z の path"""
    return LATENT_DIR / f"chemomae_latent_{split}.npy"


def get_cluster_label_path(
    split: str,
    *,
    space: Literal["latent", "ref_snv"],
    matched: bool = False,
) -> Path:
    """cluster_labels_*.npy の path"""
    base = get_split_dir(split)
    if space == "latent":
        return base / "cluster_labels_latent_ckm.npy"
    else:
        if matched:
            return base / "cluster_labels_ref_snv_ckm_matched.npy"
        return base / "cluster_labels_ref_snv_ckm.npy"


# =========================================================
# CKM / baseline 用
# =========================================================
def get_centroid_path(name: Literal["latent", "ref_snv_unmatched", "ref_snv_matched"]) -> Path:
    if name == "latent":
        return RUNS_DIR / "latent_ckm.pt"
    elif name == "ref_snv_unmatched":
        return RUNS_DIR / "ref_snv_ckm_unmatched.pt"
    elif name == "ref_snv_matched":
        return RUNS_DIR / "ref_snv_ckm_matched.pt"
    else:
        raise ValueError(f"Unknown centroid name: {name}")