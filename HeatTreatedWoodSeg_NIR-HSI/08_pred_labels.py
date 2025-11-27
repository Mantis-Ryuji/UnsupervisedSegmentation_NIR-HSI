import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import itertools
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from chemomae.clustering import CosineKMeans
from chemomae.utils.seed import set_global_seed

from src.core.config import load_config
from src.core.paths import (
    SPECTRA_DIR,
    RUNS_DIR,
    get_latent_path,
    get_reflectance_path,
    get_cluster_label_path,
    get_split_dir,
    get_centroid_path,
)
from src.preprocessing import load_sample_list
from src.viz.clusters import (
    clustering_results_list_per_sample,
    plot_cluster_distribution,
)
from src.viz.spectra import (
    plot_spectra,
    plot_spectra_2nd_derive,
)

# =========================================================
# パス定義
# =========================================================
SPLITS = ["train", "val", "test"]

OPTION_CKM_PATH = RUNS_DIR / "clustering_report" / "option_ckm.json"
LATENT_CKM_PATH = get_centroid_path("latent")
WAVE_PATH = SPECTRA_DIR / "wavenumber.npy"


# =========================================================
# ユーティリティ
# =========================================================
def _latent_path(split: str) -> Path:
    return get_latent_path(split)


def _reflectance_path(split: str) -> Path:
    return get_reflectance_path(split, snv=False, downsampled=False)


def _snv_path(split: str) -> Path:
    return get_reflectance_path(split, snv=True, downsampled=False)


def _name_list_path(split: str) -> Path:
    return get_split_dir(split) / f"{split}_name_list.json"


def _cluster_label_latent_path(split: str) -> Path:
    return get_cluster_label_path(split, space="latent", matched=False)


def load_latent(path: Path | str) -> torch.Tensor:
    """
    潜在表現(.npy)を memmap で読み、float32 Tensor にして返す。
    """
    arr = np.load(path, mmap_mode="r")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return torch.from_numpy(arr)


@torch.no_grad()
def predict_split(ckm: CosineKMeans, split: str, chunk: int, device: str) -> np.ndarray:
    """
    潜在を読み込み、CosineKMeans でクラスタ推論。
    np.ndarray (CPU) を返す。
    """
    X = load_latent(_latent_path(split)).to(device, non_blocking=True)
    labels = ckm.predict(X, return_dist=False, chunk=chunk)
    return labels.cpu().numpy()


# =========================================================
# main
# =========================================================
def main() -> None:
    # --- config ---
    cfg = load_config()
    chunk = int(cfg.clustering.chunk)
    seed = int(cfg.training.seed)

    # --- device / seed ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(seed)

    # --- k_opt の復元 ---
    with OPTION_CKM_PATH.open("r", encoding="utf-8") as f:
        option: Dict = json.load(f)
    k_opt = int(option["k_opt"])

    # --- centroid 復元（latent） ---
    ckm = CosineKMeans(
        n_components=k_opt,
        device=device,
        random_state=seed,
    )
    ckm.load_centroids(LATENT_CKM_PATH)

    # =========================================================
    # split ごとのデータ保持
    # =========================================================
    merged_results_name_list = []
    merged_results_list = []

    merged_labels_per_split = {}
    merged_ref_per_split = {}
    merged_snv_per_split = {}

    # =========================================================
    # split loop
    # =========================================================
    for split in SPLITS:

        # --- サンプル名リスト ---
        name_list_path = _name_list_path(split)
        sample_name_list = load_sample_list(str(name_list_path))
        merged_results_name_list.append(sample_name_list)

        # --- latent クラスタ予測 ---
        labels = predict_split(ckm, split, chunk, device=device)
        np.save(_cluster_label_latent_path(split), labels)
        merged_labels_per_split[split] = labels

        # --- サンプル単位の結果（mask→H×W ラベルマップ）---
        results_list_per_sample = clustering_results_list_per_sample(
            data_folder=split,
            sample_name_list=sample_name_list,
            cluster_labels=labels,
            mode="latent_ckm",
        )
        merged_results_list.append(results_list_per_sample)

        # --- スペクトル読込（後で可視化に使用） ---
        merged_ref_per_split[split] = np.load(_reflectance_path(split))
        merged_snv_per_split[split] = np.load(_snv_path(split))

    # =========================================================
    # 1) クラスタ分布可視化（train+val+test 全体）
    # =========================================================
    flat_names = list(itertools.chain.from_iterable(merged_results_name_list))
    flat_results = list(itertools.chain.from_iterable(merged_results_list))

    plot_cluster_distribution(
        results_list_per_sample=flat_results,
        sample_name_list=flat_names,
        optimal_k=k_opt,
        mode="latent_ckm",
    )

    # =========================================================
    # 2) スペクトル可視化（train + val のみ）
    # =========================================================
    ref_trainval = np.concatenate(
        [merged_ref_per_split["train"], merged_ref_per_split["val"]],
        axis=0
    )
    snv_trainval = np.concatenate(
        [merged_snv_per_split["train"], merged_snv_per_split["val"]],
        axis=0
    )
    labels_trainval = np.concatenate(
        [merged_labels_per_split["train"], merged_labels_per_split["val"]],
        axis=0
    )

    wave = np.load(WAVE_PATH)

    plot_spectra(wave, ref_trainval, snv_trainval, labels_trainval)
    plot_spectra_2nd_derive(wave, ref_trainval, labels_trainval)

    print("✅ ALL DONE")


if __name__ == "__main__":
    main()