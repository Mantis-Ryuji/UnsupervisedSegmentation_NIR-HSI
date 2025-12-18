from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Sequence, Tuple, Union, Optional
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


Connectivity = Literal[4, 8]
ArrayLike = Union[Sequence[float], np.ndarray]
SCSMetric = Literal["scs_intra", "scs_inter"]


@dataclass(frozen=True)
class SCSResult:
    """
    Spatial Consistency Score (SCS) の計算結果を保持するデータクラス。

    本クラスは、クラスタリング結果を画像平面（NIR-HSI の画素配置）
    に投影したときの **空間的一貫性（spatial coherence）** を定量化する
    指標群をまとめたものである。

    ここで評価しているのは、
    - 特徴空間（潜在空間）におけるクラスタの分離性や compactness ではなく、
    - **ラベルマップが空間的にどれだけ滑らか・連続した領域として現れているか**
    である点に注意されたい。

    Attributes
    ----------
    scs_intra : float
        近傍一致率（Spatial neighbor agreement ratio）[0, 1]。

        有効画素同士で定義される近傍辺 (i, j) に対して、
        ラベルが一致している割合を表す：

            scs_intra = mean(1[y_i == y_j])

        値が大きいほど、同一ラベルの画素が空間的にまとまっており、
        ラベルマップが「ザラザラしていない」ことを意味する。

    scs_inter : float
        近傍不一致率[0, 1]。

        scs_inter = 1 - scs_intra で定義され、
        境界の多さ（空間的な分断の度合い）を表す。

    n_edges : int
        評価に用いた近傍辺の総数。
        valid_mask や ignore_label により除外された画素間の辺は含まれない。

    n_pixels : int
        評価対象となった有効画素数。

    n_components_total : int
        全ラベルを通じた連結成分の総数（ignore_label を除外した後）。

        値が大きいほど、ラベルマップが細切れになっていることを示す。

    n_components_by_label : Dict[int, int]
        各ラベルごとの連結成分数。

        例えば、同じクラスタラベルが複数の飛び地として現れている場合、
        そのラベルに対する値が大きくなる。

    mean_component_size_by_label : Dict[int, float]
        各ラベルごとの連結成分の平均サイズ（画素数）。

        値が小さいほど、そのラベルは空間的に細かく分断されている。

    small_component_ratio : Optional[float]
        「小領域」と判定された連結成分（サイズ <= small_comp_thresh）
        に属する画素の割合。

        None の場合は、小領域率を計算していないことを意味する。
    """

    scs_intra: float
    scs_inter: float
    n_edges: int
    n_pixels: int
    n_components_total: int
    n_components_by_label: Dict[int, int]
    mean_component_size_by_label: Dict[int, float]
    small_component_ratio: Optional[float]


def compute_scs(
    label_map: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray] = None,
    ignore_label: Optional[int] = None,
    connectivity: Connectivity = 4,
    small_comp_thresh: Optional[int] = 16,
) -> SCSResult:
    r"""
    2次元ラベルマップから Spatial Consistency Score (SCS) を計算する。

    概要
    ----
    本関数は、クラスタリング結果を **画像平面（NIR-HSI の空間配置）**
    に投影した際の「空間的一貫性」を評価するための指標を計算する。

    ここでの SCS は、
    - silhouette や Davies-Bouldin index のような
      **特徴空間（潜在空間）上のクラスタ品質指標ではない**
    - ラベルマップが **面として連続しているか / 飛び地が多いか**
      という、空間的な解釈可能性を評価するための指標
    である。

    本関数は以下の2系統の情報を返す：
    1) 近傍一致率（SCS_intra）
       - ラベルマップがどれだけ滑らかか
    2) 連結成分統計
       - 同一ラベルがどれだけ分断されているか（飛び地の多さ）

    Parameters
    ----------
    label_map : np.ndarray
        形状 (H, W) の2次元配列。
        各画素に対応するクラスタラベル（整数）を与える。
        NaN は許可されない。

    valid_mask : Optional[np.ndarray], default=None
        形状 (H, W) のブール配列。
        True の画素のみを評価対象とする。

        木材領域マスクなど、
        「背景や無効領域を除外したい」場合に用いる。
        None の場合は全画素を有効とみなす。

    ignore_label : Optional[int], default=None
        指定したラベル値を評価から除外する。
        例えば背景ラベルを -1 としている場合に指定する。

        valid_mask と併用した場合は、
        両方の条件を満たす画素のみが評価対象となる。

    connectivity : {4, 8}, default=4
        近傍関係および連結成分の定義。

        - 4 : 上下左右の4近傍
        - 8 : 斜めを含む8近傍

        まずは 4 近傍を用いるのが無難であり、
        8 近傍は境界に対してより厳しい評価となる。

    small_comp_thresh : Optional[int], default=16
        「小領域」とみなす連結成分サイズ（画素数）の閾値。

        この値以下の成分に属する画素の割合を
        small_component_ratio として返す。
        None を指定した場合、この指標は計算しない。

    Returns
    -------
    SCSResult
        Spatial Consistency Score および連結成分統計をまとめた結果。

    Notes
    -----
    - SCS_intra はクラスタ数 K が増えると一般に低下しやすい
      （境界が増えるため）。
      そのため、**同一 K・同一マスク条件**での比較が前提となる。
    - SCS は「良い境界」と「ノイズ由来の境界」を区別しない。
      平均スペクトルの解釈や可視化結果と併用することが重要である。
    """

    y = np.asarray(label_map)
    if y.ndim != 2:
        raise ValueError(f"label_map must be 2D, got shape={y.shape}")
    if not np.issubdtype(y.dtype, np.integer):
        # floatラベル等を許す場合はここを緩めてもいいが、まずは事故防止で弾く
        raise TypeError(f"label_map must be integer dtype, got dtype={y.dtype}")

    H, W = y.shape

    if valid_mask is None:
        valid = np.ones((H, W), dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask)
        if valid_mask.shape != (H, W):
            raise ValueError(f"valid_mask must have shape {(H, W)}, got {valid_mask.shape}")
        valid = valid_mask.astype(bool)

    if ignore_label is not None:
        valid = valid & (y != int(ignore_label))

    n_pixels = int(valid.sum())
    if n_pixels == 0:
        return SCSResult(
            scs_intra=0.0,
            scs_inter=0.0,
            n_edges=0,
            n_pixels=0,
            n_components_total=0,
            n_components_by_label={},
            mean_component_size_by_label={},
            small_component_ratio=None if small_comp_thresh is None else 0.0,
        )

    # ---------------------------------------------------------
    # 1) SCS_intra: neighbor agreement ratio over valid edges
    # ---------------------------------------------------------
    # edges: right and down (and diagonals if 8)
    agree = 0
    edges = 0

    # right neighbors
    v = valid[:, :-1] & valid[:, 1:]
    edges_r = int(v.sum())
    if edges_r:
        agree += int((y[:, :-1][v] == y[:, 1:][v]).sum())
        edges += edges_r

    # down neighbors
    v = valid[:-1, :] & valid[1:, :]
    edges_d = int(v.sum())
    if edges_d:
        agree += int((y[:-1, :][v] == y[1:, :][v]).sum())
        edges += edges_d

    if connectivity == 8:
        # down-right diagonal
        v = valid[:-1, :-1] & valid[1:, 1:]
        edges_dr = int(v.sum())
        if edges_dr:
            agree += int((y[:-1, :-1][v] == y[1:, 1:][v]).sum())
            edges += edges_dr

        # down-left diagonal
        v = valid[:-1, 1:] & valid[1:, :-1]
        edges_dl = int(v.sum())
        if edges_dl:
            agree += int((y[:-1, 1:][v] == y[1:, :-1][v]).sum())
            edges += edges_dl

    if edges == 0:
        scs_intra = 0.0
    else:
        scs_intra = float(agree / edges)

    scs_inter = float(1.0 - scs_intra)

    # ---------------------------------------------------------
    # 2) Connected components stats per label (within valid)
    # ---------------------------------------------------------
    # BFS/DFS over valid pixels for each label
    visited = np.zeros((H, W), dtype=bool)

    # Neighbor offsets for components
    if connectivity == 4:
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    n_components_by_label: Dict[int, int] = {}
    sum_comp_sizes_by_label: Dict[int, int] = {}
    n_components_total = 0

    small_pixels = 0
    small_thresh = None if small_comp_thresh is None else int(small_comp_thresh)

    # iterate only valid pixels
    ys, xs = np.where(valid)
    for r0, c0 in zip(ys.tolist(), xs.tolist()):
        if visited[r0, c0]:
            continue
        lbl = int(y[r0, c0])

        # start new component
        n_components_total += 1
        n_components_by_label[lbl] = n_components_by_label.get(lbl, 0) + 1

        # stack-based DFS
        stack = [(r0, c0)]
        visited[r0, c0] = True
        comp_size = 0

        while stack:
            r, c = stack.pop()
            comp_size += 1
            for dr, dc in neigh:
                rr = r + dr
                cc = c + dc
                if rr < 0 or rr >= H or cc < 0 or cc >= W:
                    continue
                if not valid[rr, cc] or visited[rr, cc]:
                    continue
                if int(y[rr, cc]) != lbl:
                    continue
                visited[rr, cc] = True
                stack.append((rr, cc))

        sum_comp_sizes_by_label[lbl] = sum_comp_sizes_by_label.get(lbl, 0) + comp_size
        if small_thresh is not None and comp_size <= small_thresh:
            small_pixels += comp_size

    mean_component_size_by_label: Dict[int, float] = {}
    for lbl, ncomp in n_components_by_label.items():
        mean_component_size_by_label[lbl] = float(sum_comp_sizes_by_label[lbl] / ncomp)

    small_component_ratio = None
    if small_thresh is not None:
        small_component_ratio = float(small_pixels / n_pixels)

    return SCSResult(
        scs_intra=scs_intra,
        scs_inter=scs_inter,
        n_edges=int(edges),
        n_pixels=n_pixels,
        n_components_total=int(n_components_total),
        n_components_by_label=n_components_by_label,
        mean_component_size_by_label=mean_component_size_by_label,
        small_component_ratio=small_component_ratio,
    )


def _as_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={x.shape}")
    if x.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return x


def _mean_std(x: np.ndarray) -> tuple[float, float]:
    m = float(np.mean(x))
    if x.size < 2:
        return m, 0.0
    return m, float(np.std(x, ddof=1))


def _extract_metric_per_split(
    scores: Dict[str, Dict[str, list[Dict[str, Any]]]],
    *,
    space: Literal["ref_snv", "latent"],
    split: str,
    metric: SCSMetric,
) -> np.ndarray:
    """
    scores[space][split] は、各サンプルの dict の list を想定。
    例: scores["latent"]["train"] = [{"sample":..., "scs_intra":0.9, ...}, ...]
    """
    if space not in scores:
        raise KeyError(f"Missing '{space}' in scores keys={list(scores.keys())}")
    if split not in scores[space]:
        raise KeyError(f"Missing split '{split}' in scores['{space}']")

    rows = scores[space][split]
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError(f"scores['{space}']['{split}'] must be non-empty list")

    vals = []
    for r in rows:
        if metric not in r:
            raise KeyError(f"Missing '{metric}' in a row: keys={list(r.keys())}")
        v = r[metric]
        if v is None:
            continue
        vals.append(float(v))

    return _as_1d_float(np.asarray(vals, dtype=float), f"{space}:{split}:{metric}")


def plot_scs_bar(
    *,
    scores: Dict[str, Dict[str, list[Dict[str, Any]]]],
    save_path: Union[str, Path],
    metric: SCSMetric = "scs_intra",
    splits: Tuple[str, str, str] = ("train", "val", "test"),
    labels: Tuple[str, str] = ("ref_snv", "latent"),
    colors: Tuple[str, str] = ("tab:blue", "tab:orange"),
    ylabel: str | None = None,
    y_min: float = 0.0,
    y_max: float = 1.0,
    dpi: int = 200,
) -> None:
    """
    Split-wise SCS bar chart (ref_snv vs latent).

    Parameters
    ----------
    scores : dict
        13_compare_scs.py が吐く辞書（JSONを load したもの）を想定。
        scores["ref_snv"][split] / scores["latent"][split] は list[dict]。
    metric : {"scs_intra","scs_inter"}
        描画する指標。
    Notes
    -----
    - std は「各サンプルの SCS の split 内ばらつき（サンプル標準偏差）」。
    - テキストはエラーバー上端に mean±std (.3g)。
    """
    if ylabel is None:
        ylabel = f"SCS ({metric})"

    # ---- compute mean/std per split ----
    ref_means, ref_stds = [], []
    lat_means, lat_stds = [], []
    for sp in splits:
        r = _extract_metric_per_split(scores, space="ref_snv", split=sp, metric=metric)
        l = _extract_metric_per_split(scores, space="latent", split=sp, metric=metric)

        rm, rs = _mean_std(r)
        lm, ls = _mean_std(l)

        ref_means.append(rm); ref_stds.append(rs)
        lat_means.append(lm); lat_stds.append(ls)

    ref_means = np.asarray(ref_means, dtype=float)
    ref_stds  = np.asarray(ref_stds, dtype=float)
    lat_means = np.asarray(lat_means, dtype=float)
    lat_stds  = np.asarray(lat_stds, dtype=float)

    # ---- plot ----
    n = len(splits)
    x = np.arange(n, dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=dpi)

    bars_ref = ax.bar(
        x - width / 2,
        ref_means,
        yerr=ref_stds,
        capsize=5,
        width=width,
        color=colors[0],
        align="center",
        label=labels[0],
    )
    bars_lat = ax.bar(
        x + width / 2,
        lat_means,
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
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="lower left")

    # annotations at errorbar top: mean±std
    pad = 0.02 * (y_max - y_min + 1e-12)
    for rect, m, s in zip(bars_ref, ref_means, ref_stds):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            m + s + pad,
            f"{m:.3g}±{s:.3g}",
            ha="center",
            va="bottom",
            fontsize=9,
            clip_on=False,
        )
    for rect, m, s in zip(bars_lat, lat_means, lat_stds):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            m + s + pad,
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