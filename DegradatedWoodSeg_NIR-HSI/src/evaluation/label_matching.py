from __future__ import annotations

from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from scipy.optimize import linear_sum_assignment


def compute_label_map(y_ref: np.ndarray, y_lat: np.ndarray) -> Tuple[Dict[int, int], np.ndarray]:
    """
    ref_snv_ckm のラベルを latent_ckm に揃える置換を Hungarian で求める。
    行 = ref, 列 = latent の一致数行列を最大化。
    """
    ref_classes = np.unique(y_ref)
    lat_classes = np.unique(y_lat)
    assert ref_classes.size == lat_classes.size, "Kは共通前提（サイズ不一致）"
    K = ref_classes.size

    # 混同行列 C[r, c] = count( y_ref == r and y_lat == c )
    C = np.zeros((K, K), dtype=np.int64)
    for r in ref_classes:
        mask_r = (y_ref == r)
        col, cnt = np.unique(y_lat[mask_r], return_counts=True)
        C[r, col] = cnt

    ridx, cidx = linear_sum_assignment(-C)  # 一致数を最大化
    label_map = {int(r): int(c) for r, c in zip(ridx, cidx)}
    return label_map, C


def apply_map(y_ref: np.ndarray, label_map: Dict[int, int]) -> np.ndarray:
    y_new = y_ref.copy()
    for r, c in label_map.items():
        y_new[y_ref == r] = c
    return y_new


def plot_confusion_heatmap(C: np.ndarray, label_map: Dict[int, int], out_path: Path) -> None:
    """
    train+val の混同行列をヒートマップ表示し、
    Hungarian で選ばれたセルを枠で強調して保存。
    """
    out_path = Path(out_path)
    K = C.shape[0]
    matched = sum(C[r, c] for r, c in label_map.items())
    coverage = matched / C.sum() if C.sum() > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    im = ax.imshow(C, aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="count")

    ax.set_xlabel("Latent label")
    ax.set_ylabel("Ref(SNV) label")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))

    # 対応セルを枠で強調
    for r, c in label_map.items():
        ax.add_patch(
            Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, linewidth=2.0, edgecolor="white")
        )

    ax.set_title(f"Confusion (train+val)  |  matched coverage = {coverage:.3g}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_aligned_ref_centroids(
    path_unmatched: Path,
    path_out: Path,
    label_map: Dict[int, int],
) -> None:
    """
    ref(行=refラベル順) の centroid を、
    label_map(ref->latent) に基づき行を latent 順 (0..K-1) に並べ替えて保存する。
    """
    path_unmatched = Path(path_unmatched)
    path_out = Path(path_out)

    obj = torch.load(path_unmatched, map_location="cpu")
    C = obj["centroids"] if isinstance(obj, dict) and "centroids" in obj else obj
    if isinstance(C, torch.Tensor):
        C = C.detach().cpu().numpy()
    C = np.asarray(C, dtype=np.float32)
    if C.ndim != 2:
        raise ValueError(f"centroids must be 2D, got {C.shape}")

    # 念のため行方向に正規化
    n = np.linalg.norm(C, axis=1, keepdims=True)
    C = C / np.clip(n, 1e-12, None)

    K = C.shape[0]
    # inverse: latent -> ref を作り latent 順に並べ替え
    lat2ref = {int(lat): int(ref) for ref, lat in label_map.items()}
    C_aligned = np.empty_like(C)
    for lat_idx in range(K):
        if lat_idx not in lat2ref:
            raise KeyError(f"latent index {lat_idx} is missing in label_map")
        ref_idx = lat2ref[lat_idx]
        C_aligned[lat_idx] = C[ref_idx]

    # 再正規化して保存（dict形式で揃える）
    n = np.linalg.norm(C_aligned, axis=1, keepdims=True)
    C_aligned = C_aligned / np.clip(n, 1e-12, None)

    path_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"centroids": torch.from_numpy(C_aligned)}, path_out)


def verify_label_matching(y_ref_matched: np.ndarray, y_latent: np.ndarray) -> None:
    """
    ハンガリアン整合後のラベル集合が一致しているか（K と ID 集合）を検証。
    """
    U_ref = np.unique(y_ref_matched)
    U_lat = np.unique(y_latent)
    assert U_ref.size == U_lat.size, f"K mismatch: ref={U_ref.size}, latent={U_lat.size}"
    assert np.array_equal(np.sort(U_ref), np.sort(U_lat)), "Label sets differ; not matched?"
    print(f"[OK] Labels matched. K={U_ref.size}")