from __future__ import annotations

from typing import List, Literal
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap
import colorcet as cc

from ..core.paths import DATA_DIR, IMAGES_DIR
from ..preprocessing.io_mask import _save_tight_image


# =============================================================
# グローバルパス（ラベル画像・マスク）
# =============================================================
LABEL_IMG_DIR = IMAGES_DIR / "labels"


def _mask_path(data_folder: str, name: str) -> Path:
    return DATA_DIR / data_folder / "masks" / f"{name}_mask.npy"


def _label_npy_path(data_folder: str, name: str, mode: str) -> Path:
    return DATA_DIR / data_folder / "labels" / f"{name}_cluster_labels_{mode}.npy"


def _label_png_path(name: str, mode: str) -> Path:
    return LABEL_IMG_DIR / f"{name}_cluster_labels_{mode}.png"


# =============================================================
# クラスタラベル -> 画像再構成 & 可視化
# =============================================================
def clustering_results_list_per_sample(
    data_folder: str,
    sample_name_list: List[str],
    cluster_labels: np.ndarray,
    mode: Literal["ref_snv_ckm_matched", "latent_ckm"] = "latent_ckm",
) -> List[np.ndarray]:
    """
    各サンプルに対してクラスタリング結果をマスクに基づいて再構築し、
    元画像サイズ (H×W) のラベルマップを作成・保存する。
    """
    mask_list = []
    for name in sample_name_list:
        mask = np.load(_mask_path(data_folder, name))
        mask_list.append(mask)

    # --- マスク連結と整合性チェック ---
    cat_mask = np.concatenate(mask_list, axis=0)
    n_valid_pixels = int(np.sum(cat_mask == 1))
    if len(cluster_labels) != n_valid_pixels:
        raise ValueError(
            f"cluster_labels の数 {len(cluster_labels)} が、mask==1 の総数 {n_valid_pixels} と一致しません"
        )

    # --- 各サンプルごとの再構成 ---
    results_list_per_sample = []
    cluster_offset = 0

    # 保存先ラベルディレクトリ
    label_dir = DATA_DIR / data_folder / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(sample_name_list):
        mask = mask_list[i]
        h, w = mask.shape
        size = h * w

        mask_flat = mask.flatten()
        cluster_labels_this_sample = np.full(size, -1)
        valid_pixels = (mask_flat == 1)
        num_valid = int(np.sum(valid_pixels))

        cluster_labels_this_sample[valid_pixels] = cluster_labels[
            cluster_offset : cluster_offset + num_valid
        ]
        cluster_offset += num_valid

        result = cluster_labels_this_sample.reshape(h, w)
        np.save(_label_npy_path(data_folder, name, mode), result)
        results_list_per_sample.append(result)

    return results_list_per_sample


def get_glasbey_with_white(n_clusters: int) -> ListedColormap:
    """
    Glasbeyカラーマップ（視覚的に識別しやすい離散色）に
    白（背景色）を先頭に追加して返す。
    """
    return ListedColormap(["white"] + list(cc.glasbey_light[:n_clusters]))


def plot_cluster_distribution(
    results_list_per_sample: List[np.ndarray],
    sample_name_list: List[str],
    optimal_k: int,
    mode: Literal["ref_snv_ckm", "latent_ckm"] = "latent_ckm",
    width_inch: float = 8.0,
):
    """
    各サンプルのクラスタ割り当てマップ（形状: H×W）を可視化して保存する。
    """
    vmin = -1
    vmax = optimal_k - 1
    cmap = get_glasbey_with_white(optimal_k)
    ticks = list(range(vmin, vmax + 1))

    for label, name in zip(results_list_per_sample, sample_name_list):
        _save_tight_image(
            img=label,
            out_path=_label_png_path(name, mode),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            width_inch=width_inch,
            add_cbar=True,
            cbar_label="Cluster ID",
            cbar_ticks=ticks,
        )