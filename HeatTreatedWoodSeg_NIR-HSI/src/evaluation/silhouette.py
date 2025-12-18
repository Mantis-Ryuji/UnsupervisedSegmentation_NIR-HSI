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
    ref_scores: dict[str, ArrayLike],
    ref_cluster_ids: dict[str, ArrayLike],
    latent_scores: dict[str, ArrayLike],
    latent_cluster_ids: dict[str, ArrayLike],
    save_path: Union[str, Path],
    splits: Tuple[str, str, str] = ("train", "val", "test"),
    labels: Tuple[str, str] = ("ref_snv", "latent"),
    colors: Tuple[str, str] = ("tab:blue", "tab:orange"),
    ylabel: str = "Silhouette score (cosine)",
    dpi: int = 200,
) -> None:
    """
    Split-wise silhouette bar chart.

    Parameters
    ----------
    ref_scores, latent_scores : dict[str, ArrayLike]
        各 split の **各点シルエット**（s_i）の 1D 配列。
        例: {"train": s_train, "val": s_val, "test": s_test}
    ref_cluster_ids, latent_cluster_ids : dict[str, ArrayLike]
        各 split の **各点クラスタID**（1D int 配列）。
        std は「クラスタ平均シルエット {mean_k} の split 内ばらつき」で計算する。
    splits : tuple[str, str, str]
        描画順。デフォルトは ("train","val","test")。

    Notes
    -----
    - 棒は [-1, 1] の軸で表示（bottom=-1 からの高さで描画）。
    - テキストはエラーバー上端に mean±std (.3g)。
    """
    # ---- validate splits presence ----
    for sp in splits:
        if sp not in ref_scores or sp not in ref_cluster_ids:
            raise KeyError(f"Missing split '{sp}' in ref_scores/ref_cluster_ids")
        if sp not in latent_scores or sp not in latent_cluster_ids:
            raise KeyError(f"Missing split '{sp}' in latent_scores/latent_cluster_ids")

    # ---- compute mean/std per split ----
    ref_means, ref_stds = [], []
    lat_means, lat_stds = [], []
    for sp in splits:
        rs = _to_1d(ref_scores[sp], f"ref_scores[{sp}]")
        rc = _to_1d_int(ref_cluster_ids[sp], f"ref_cluster_ids[{sp}]")
        ls = _to_1d(latent_scores[sp], f"latent_scores[{sp}]")
        lc = _to_1d_int(latent_cluster_ids[sp], f"latent_cluster_ids[{sp}]")

        rm, rstd = _cluster_mean_std(rs, rc)
        lm, lstd = _cluster_mean_std(ls, lc)

        ref_means.append(rm); ref_stds.append(rstd)
        lat_means.append(lm); lat_stds.append(lstd)

    ref_means = np.asarray(ref_means, dtype=float)
    ref_stds  = np.asarray(ref_stds, dtype=float)
    lat_means = np.asarray(lat_means, dtype=float)
    lat_stds  = np.asarray(lat_stds, dtype=float)

    # ---- plot ----
    n = len(splits)
    x = np.arange(n, dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=dpi)

    bottom = -1.0
    ref_heights = ref_means - bottom
    lat_heights = lat_means - bottom

    bars_ref = ax.bar(
        x - width / 2,
        ref_heights,
        bottom=bottom,
        yerr=ref_stds,
        capsize=5,
        width=width,
        color=colors[0],
        align="center",
        label=labels[0],
    )
    bars_lat = ax.bar(
        x + width / 2,
        lat_heights,
        bottom=bottom,
        yerr=lat_stds,
        capsize=5,
        width=width,
        color=colors[1],
        align="center",
        label=labels[1],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(list(splits))
    ax.set_ylabel(ylabel)
    ax.set_ylim(-1.0, 1.0)
    ax.legend()

    # annotations at errorbar top: mean±std
    for rect, m, s in zip(bars_ref, ref_means, ref_stds):
        y = m + s
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            y + 0.02,
            f"{m:.3g}±{s:.3g}",
            ha="center",
            va="bottom",
            fontsize=9,
            clip_on=False,
        )
    for rect, m, s in zip(bars_lat, lat_means, lat_stds):
        y = m + s
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            y + 0.02,
            f"{m:.3g}±{s:.3g}",
            ha="center",
            va="bottom",
            fontsize=9,
            clip_on=False,
        )

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)