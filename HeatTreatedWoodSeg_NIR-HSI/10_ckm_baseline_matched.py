import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import itertools
from pathlib import Path

import numpy as np
import torch

from chemomae.clustering import CosineKMeans
from chemomae.utils.seed import set_global_seed

from src.core.config import load_config
from src.core.paths import (
    RUNS_DIR,
    IMAGES_DIR,
    get_reflectance_path,
    get_cluster_label_path,
    get_split_dir,
)
from src.preprocessing import load_sample_list
from src.viz import (
    clustering_results_list_per_sample,
    plot_cluster_distribution,
)
from src.evaluation.label_matching import (
    compute_label_map,
    apply_map,
    plot_confusion_heatmap,
    save_aligned_ref_centroids
)

# =========================================================
# パス定義
# =========================================================
CENTROID_UNMATCHED_PATH = RUNS_DIR / "ref_snv_ckm_unmatched.pt"
CENTROID_MATCHED_PATH   = RUNS_DIR / "ref_snv_ckm_matched.pt"

IMG_LOG_DIR = IMAGES_DIR / "logs"
CONF_HEATMAP_PNG = IMG_LOG_DIR / "label_map_confusion_train.png"


# レポート系は runs/clustering_report/ 配下
REPORT_DIR       = RUNS_DIR / "clustering_report"
OPTION_CKM_PATH  = REPORT_DIR / "option_ckm.json"
MAP_JSON_PATH    = REPORT_DIR / "label_matching_ref_snv_to_latent.json"


# =========================================================
# ヘルパー: パスラッパ
# =========================================================
def _reflectance_snv_path(split: str) -> Path:
    return get_reflectance_path(split, snv=True, downsampled=False)


def _ref_label_path(split: str) -> Path:
    # baseline（ref_snv_ckm）のラベル
    return get_cluster_label_path(split, space="ref_snv", matched=False)


def _ref_label_matched_path(split: str) -> Path:
    # latent に揃えた baseline ラベル
    return get_cluster_label_path(split, space="ref_snv", matched=True)


def _latent_label_path(split: str) -> Path:
    # latent_ckm のラベル
    return get_cluster_label_path(split, space="latent", matched=False)


def _name_list_path(split: str) -> Path:
    return get_split_dir(split) / f"{split}_name_list.json"


# =========================================================
# ユーティリティ
# =========================================================
def _load_float32(path: Path) -> torch.Tensor:
    """Load a numpy array as float32 torch.Tensor (CPU).

    Notes
    -----
    - mmap_mode='r' で読み込み、必要なら float32 にキャストする。
    """
    x = np.load(path, mmap_mode="r")
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return torch.from_numpy(x)


@torch.no_grad()
def _predict_split(ckm: CosineKMeans, split: str, chunk: int, device: str) -> np.ndarray:
    """
    ref_snv 空間の SNV スペクトルに対して CosineKMeans のラベルを付与。
    """
    X_path = _reflectance_snv_path(split)
    X = np.load(X_path, mmap_mode="r")
    X_tensor = torch.from_numpy(X).to(device, non_blocking=True)
    labels = ckm.predict(X_tensor, return_dist=False, chunk=chunk)
    return labels.cpu().numpy()


# =========================================================
# main
# =========================================================
def main() -> None:
    # --- config 読み込み ---
    cfg = load_config()
    clustering_cfg = cfg.clustering
    training_cfg = cfg.training

    chunk = int(clustering_cfg.chunk)
    seed = int(training_cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(seed)

    # --- ディレクトリ作成 ---
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    IMG_LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --- latent CKM の k_opt を読み込み ---
    with OPTION_CKM_PATH.open("r", encoding="utf-8") as f:
        option = json.load(f)
    k_opt = int(option["k_opt"])

    # --- ref_snv 空間で CKM を学習（train のみ） ---
    X_train = _load_float32(_reflectance_snv_path("train"))
    ckm = CosineKMeans(n_components=k_opt, device=device, random_state=seed)
    ckm.fit(X_train, chunk=chunk)
    ckm.save_centroids(CENTROID_UNMATCHED_PATH)

    # --- baseline 側 (ref_snv) のラベル付け ---
    for split in ["train", "val", "test"]:
        labels = _predict_split(ckm, split, chunk, device=device)
        np.save(_ref_label_path(split), labels)

    # --- latent 側ラベルの存在チェック ---
    for split in ["train", "val", "test"]:
        if not _latent_label_path(split).exists():
            raise FileNotFoundError(
                f"latent側の {split} ラベルが見つかりません: {_latent_label_path(split)}\n"
                "先に ChemoMAE latent へ CKM を適用し、cluster_labels_latent_ckm.npy を生成してください。"
            )

    # --- train のみでラベルマップ推定 (Hungarian) ---
    y_ref_train = np.load(_ref_label_path("train"))
    y_lat_train = np.load(_latent_label_path("train"))

    label_map, confmat_train = compute_label_map(y_ref_train, y_lat_train)

    # --- ラベルマップ + 混同行列を JSON 保存 ---
    MAP_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MAP_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {"map": label_map, "confusion_train": confmat_train.tolist()},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --- 混同行列ヒートマップ（Hungarian 対応セルを枠で強調） ---
    plot_confusion_heatmap(confmat_train, label_map, CONF_HEATMAP_PNG)

    # --- ref の centroid を latent 順に整列して保存 ---
    save_aligned_ref_centroids(
        CENTROID_UNMATCHED_PATH,
        CENTROID_MATCHED_PATH,
        label_map,
    )

    # --- baseline ラベルを latent 番号に整合させた npy を split ごとに保存 ---
    for split in ["train", "val", "test"]:
        y_ref = np.load(_ref_label_path(split))
        y_ref_matched = apply_map(y_ref, label_map)
        np.save(_ref_label_matched_path(split), y_ref_matched)

    # --- サンプル単位の分布集計 & 可視化 ---
    results_list = []
    results_name_list = []
    for split in ["train", "val", "test"]:
        name_json = _name_list_path(split)
        sample_name_list = load_sample_list(str(name_json))
        results_name_list.append(sample_name_list)

        y_ref_matched = np.load(_ref_label_matched_path(split))
        results_list_per_sample = clustering_results_list_per_sample(
            data_folder=split,
            sample_name_list=sample_name_list,
            cluster_labels=y_ref_matched,
            mode="ref_snv_ckm_matched",
        )
        results_list.append(results_list_per_sample)

    merged_results_name_list = list(itertools.chain.from_iterable(results_name_list))
    merged_results_list = list(itertools.chain.from_iterable(results_list))

    plot_cluster_distribution(
        results_list_per_sample=merged_results_list,
        sample_name_list=merged_results_name_list,
        optimal_k=k_opt,
        mode="ref_snv_ckm_matched",
    )

    print("✅ ALL DONE")


if __name__ == "__main__":
    main()