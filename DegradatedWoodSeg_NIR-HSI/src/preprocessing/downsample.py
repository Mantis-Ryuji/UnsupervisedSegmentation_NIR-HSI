from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from chemomae.preprocessing import SNVScaler, cosine_fps_downsample
from ..core.paths import get_split_dir


@torch.no_grad()
def trim_rim_by_knn_cosine_gpu(
    X_np: np.ndarray,
    *,
    k: int = 50,            # 近傍数 (1 < k < M)
    q: float = 1.0,         # 下位q%（疎）を除去
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    ensure_keep: int = 1,   # 最低保持数の保険
) -> np.ndarray:
    """
    CUDA & cos類似度に基づく縁トリミング（Rim trimming）。

    距離計算は L2 正規化ベクトルで行うが、
    返り値は「元スケールの X_np からインデックス抽出したもの」を返す。
    """
    if X_np.ndim != 2:
        raise ValueError(f"X_np must be 2D, got shape={X_np.shape}")

    M, C = X_np.shape
    if k <= 1 or k >= M:
        raise ValueError(f"k must satisfy 1 < k < M, got k={k}, M={M}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # 元スケールのテンソル（戻り値用）
    X = torch.as_tensor(X_np, device=device, dtype=dtype)

    # 類似度計算用に L2 正規化したコピーを作る
    X_norm = torch.nn.functional.normalize(X, p=2, dim=1)

    # cos 類似度行列（X_norm 同士）
    sim = X_norm @ X_norm.T
    sim.fill_diagonal_(-1.0)

    # k近傍類似度
    knn_sim, _ = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)
    knn_dist = 1.0 - knn_sim  # cos 距離

    mean_knn = knn_dist.mean(dim=1)

    if q <= 0:
        keep = torch.ones(M, dtype=torch.bool, device=device)
    else:
        q = float(q)
        threshold = torch.quantile(mean_knn, 1.0 - q / 100.0)
        keep = mean_knn <= threshold

    num_keep = int(keep.sum().item())
    if num_keep == 0:
        idx = torch.argmin(mean_knn).unsqueeze(0)
    elif num_keep < ensure_keep:
        topk_idx = torch.topk(-mean_knn, k=ensure_keep, largest=True).indices
        idx = topk_idx
    else:
        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)

    # 元スケールの X からインデックス抽出して返す
    X_trim = X[idx].detach().cpu().numpy()
    return X_trim


def return_downsampled_dataset_np(
    data_folder: str,
    data_list: List[np.ndarray],   # 各: (H, W, C)
    masks: List[np.ndarray],       # 各: (H, W)  1=有効
    *,
    ratio: float = 0.1,
    seed: Optional[int] = 42,
    out_name: str = "reflectance_snv_downsampled.npy",
    trim: bool = False,
) -> np.ndarray:
    """
    SNV → マスク抽出 → FPS（cos幾何, CUDA自動使用）→ (任意)縁トリム → 連結＆保存
    保存先: data/{data_folder}/{out_name}
    """
    if len(data_list) != len(masks):
        raise ValueError(f"data_list と masks の長さが一致しません: {len(data_list)} vs {len(masks)}")

    scaler = SNVScaler()

    down_list: List[np.ndarray] = []

    for data, mask in tqdm(
        zip(data_list, masks),
        total=len(data_list),
        desc=f"Downsampling ({data_folder})",
    ):
        H, W, C = data.shape
        flat_data = data.reshape(-1, C)
        flat_mask = mask.reshape(-1)

        selected = flat_mask == 1
        if not np.any(selected):
            tqdm.write("Warning: no wood pixels in a sample, skipping.")
            continue

        X = flat_data[selected]  # (N, C)

        # SNV（ピクセル単位）
        X_snv = scaler.transform(X)

        # FPS（cosine geometry, GPU自動選択）
        if ratio >= 1.0:
            X_down = X_snv.astype(np.float32, copy=False)
        else:
            X_down = cosine_fps_downsample(
                X_snv.astype(np.float32, copy=False),
                ratio=ratio,
                seed=seed
            )

        # Rim trimming (optional)
        if trim:
            X_down = trim_rim_by_knn_cosine_gpu(X_down)

        down_list.append(X_down)

    out = (
        np.empty((0, data_list[0].shape[-1]), dtype=np.float32)
        if not down_list
        else np.concatenate(down_list, axis=0)
    )

    out_dir = get_split_dir(data_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / out_name, out)
    return out