from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt

ArrayLike = Union[Sequence[float], np.ndarray]


def _to_1d(x: ArrayLike, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={x.shape}")
    if x.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return x


def _to_1d_int(x: ArrayLike, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={x.shape}")
    if x.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.issubdtype(x.dtype, np.integer):
        # floatで来た場合も許す（例: torch -> numpy で int64 以外）
        x = x.astype(int)
    return x


def _cluster_mean_std(scores: np.ndarray, cluster_ids: np.ndarray) -> tuple[float, float]:
    """
    scores: 各点のシルエット s_i
    cluster_ids: 各点のクラスタID
    戻り値:
      mean = 全点平均 mean(s_i)
      std  = クラスタ平均 {mean_k} の標準偏差 std(mean_k)
    """
    if scores.shape != cluster_ids.shape:
        raise ValueError(f"scores and cluster_ids must have same shape, got {scores.shape} vs {cluster_ids.shape}")

    m = float(np.mean(scores))

    uniq = np.unique(cluster_ids)
    # 各クラスタの平均シルエット
    per_cluster_means = []
    for k in uniq:
        mask = (cluster_ids == k)
        if np.any(mask):
            per_cluster_means.append(float(np.mean(scores[mask])))

    if len(per_cluster_means) < 2:
        s = 0.0  # クラスタが1個以下なら“ばらつき”は定義しづらいので0扱い
    else:
        s = float(np.std(per_cluster_means, ddof=1))  # サンプル標準偏差

    return m, s


def plot_silhouette_bar(
    *,
    ref_scores: ArrayLike,
    ref_cluster_ids: ArrayLike,
    latent_scores: ArrayLike,
    latent_cluster_ids: ArrayLike,
    save_path: Union[str, Path],
    labels: Tuple[str, str] = ("ref_snv", "latent"),
    colors: Tuple[str, str] = ("tab:blue", "tab:orange"),
    ylabel: str = "Silhouette score",
    dpi: int = 200,
) -> None:
    """
    mean を棒の上端、std をエラーバーとする棒グラフを描画。
    ここで std は「クラスタごとの平均シルエット」のばらつき。

    表示テキストはエラーバー上端に mean±std (.3g)。
    """
    ref_scores = _to_1d(ref_scores, "ref_scores")
    latent_scores = _to_1d(latent_scores, "latent_scores")
    ref_cluster_ids = _to_1d_int(ref_cluster_ids, "ref_cluster_ids")
    latent_cluster_ids = _to_1d_int(latent_cluster_ids, "latent_cluster_ids")

    ref_mean, ref_std = _cluster_mean_std(ref_scores, ref_cluster_ids)
    lat_mean, lat_std = _cluster_mean_std(latent_scores, latent_cluster_ids)

    means = np.array([ref_mean, lat_mean], dtype=float)
    stds  = np.array([ref_std,  lat_std ], dtype=float)

    x = np.arange(2)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)

    bottom = -1.0
    heights = means - bottom  # = means + 1

    bars = ax.bar(
        x,
        heights,
        bottom=bottom,
        yerr=stds,
        capsize=5,
        width=0.6,
        color=colors,
        align="center",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-1.0, 1.0)

    # 注釈：エラーバー上端に mean±std
    for rect, m, s in zip(bars, means, stds):
        y = m + s
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            y + 0.02,
            f"{m:.3g}±{s:.3g}",
            ha="center",
            va="bottom",
            fontsize=10,
            clip_on=False,
        )

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)