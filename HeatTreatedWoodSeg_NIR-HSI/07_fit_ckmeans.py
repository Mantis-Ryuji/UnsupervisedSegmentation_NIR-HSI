import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from chemomae.clustering import CosineKMeans, elbow_ckmeans, plot_elbow_ckm, silhouette_score_cosine_gpu
from chemomae.utils.seed import set_global_seed

from src.core.config import load_config
from src.core.paths import (
    LATENT_DIR,
    RUNS_DIR,
    IMAGES_DIR,
    get_latent_path,
    get_centroid_path,
)


# =========================================================
# パス定義
# =========================================================
LATENT_TRAIN_PATH = get_latent_path("train")
LATENT_VAL_PATH   = get_latent_path("val")
LATENT_TEST_PATH  = get_latent_path("test")

IMG_LOG_DIR       = IMAGES_DIR / "logs"
ELBOW_IMG_PATH    = IMG_LOG_DIR / "elbow_k.png"

CENTROID_OUT_PATH = get_centroid_path("latent")         # runs/latent_ckm.pt
REPORT_DIR        = RUNS_DIR / "clustering_report"
OPTION_JSON_PATH  = REPORT_DIR / "option_ckm.json"


# =========================================================
# ユーティリティ
# =========================================================
def _load_latent(path: Path | str) -> torch.Tensor:
    """
    学習済みChemoMAEで抽出した潜在表現を読み込む関数。

    - np.load(mmap_mode="r") でメモリ節約しつつ読み込み
    - dtype を float32 に統一して torch.Tensor 化
    """
    x = np.load(path, mmap_mode="r")
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return torch.from_numpy(x)


@torch.no_grad()
def _assign_by_centroids(latent: torch.Tensor, centroids: torch.Tensor, *, device: str, chunk: int) -> torch.Tensor:
    """
    重心（K,d）に対して cosine 最近傍で割り当てラベル（N,）を返す。
    """
    X = latent.to(device)
    C = centroids.to(device)

    X = F.normalize(X, dim=1)
    C = F.normalize(C, dim=1)

    N = X.shape[0]
    labels = torch.empty(N, dtype=torch.long, device=device)

    for i0 in range(0, N, chunk):
        i1 = min(N, i0 + chunk)
        sims = X[i0:i1] @ C.T  # (b,K)
        labels[i0:i1] = sims.argmax(dim=1)

    return labels


@torch.no_grad()
def _predict_labels(ckm: CosineKMeans, latent: torch.Tensor, *, device: str, chunk: int) -> torch.Tensor:
    """
    可能なら ckm.predict を使い、無ければ内部重心で最近傍割り当て。
    """
    if hasattr(ckm, "predict") and callable(getattr(ckm, "predict")):
        # 実装が対応している場合
        return ckm.predict(latent.to(device), chunk=chunk)

    # よくある属性名に対応
    if hasattr(ckm, "centroids"):
        centroids = getattr(ckm, "centroids")
    elif hasattr(ckm, "centroids_"):
        centroids = getattr(ckm, "centroids_")
    else:
        raise AttributeError("CosineKMeans has neither predict() nor centroids/centroids_ attribute.")

    return _assign_by_centroids(latent, centroids, device=device, chunk=chunk)


def main() -> None:
    # --- config 読み込み ---
    cfg = load_config()
    cl_cfg = cfg.clustering   # ClusteringConfig（k_max, chunk など）
    tr_cfg = cfg.training     # TrainingConfig（seed など）

    k_max = int(cl_cfg.k_max)      # 最大クラスタ数（エルボー法探索範囲）
    chunk = int(cl_cfg.chunk)      # 大規模データ時の分割処理サイズ

    # --- device / seed ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(tr_cfg.seed)
    set_global_seed(seed)

    # --- 出力ディレクトリを作成 ---
    IMG_LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LATENT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # 潜在表現の読み込み（trainのみでfitする）
    # =====================================================
    latent_train = _load_latent(LATENT_TRAIN_PATH)
    latent_val   = _load_latent(LATENT_VAL_PATH)
    latent_test  = _load_latent(LATENT_TEST_PATH)

    # =====================================================
    # クラスタ数 K を自動決定（曲率法 / エルボー法）: trainのみ
    # =====================================================
    k_list, inertias, k_elbow, idx, kappa = elbow_ckmeans(
        CosineKMeans,   # クラスタリングアルゴリズム（ここではCosineKMeans）
        latent_train,
        device=device,
        k_max=k_max,
        chunk=chunk,
        verbose=True,
        random_state=seed,
    )

    # --- エルボーカーブを保存 ---
    plot_elbow_ckm(k_list, inertias, k_elbow, idx)
    plt.savefig(ELBOW_IMG_PATH, dpi=300)
    plt.close()

    # =====================================================
    # k_opt はエルボー法で決定（silhouette は使用しない）
    # =====================================================
    k_opt = int(k_elbow)

    # =====================================================
    # CosineKMeans による実クラスタリング: trainのみ
    # =====================================================
    ckm = CosineKMeans(
        n_components=int(k_opt),
        device=device,
        random_state=seed,
    )
    ckm.fit(latent_train, chunk=chunk)

    # --- 学習済みクラスタ中心を保存 ---
    ckm.save_centroids(CENTROID_OUT_PATH)

    # =====================================================
    # trainで学習した重心に対して val/test を割り当てて検証（ログ出力のみ）
    # =====================================================
    labels_train = _predict_labels(ckm, latent_train, device=device, chunk=chunk)
    labels_val   = _predict_labels(ckm, latent_val,   device=device, chunk=chunk)
    labels_test  = _predict_labels(ckm, latent_test,  device=device, chunk=chunk)

    sil_train = silhouette_score_cosine_gpu(
        latent_train.to(device), labels_train.to(device),
        device=device, chunk=chunk, return_numpy=True
    )
    sil_val = silhouette_score_cosine_gpu(
        latent_val.to(device), labels_val.to(device),
        device=device, chunk=chunk, return_numpy=True
    )
    sil_test = silhouette_score_cosine_gpu(
        latent_test.to(device), labels_test.to(device),
        device=device, chunk=chunk, return_numpy=True
    )

    print("===== Silhouette Evaluation (fixed centroids from train) =====")
    print(f"  train: {float(sil_train):.6f}")
    print(f"  val  : {float(sil_val):.6f}")
    print(f"  test : {float(sil_test):.6f}")
    print()

    # ---- クラスタリングの設定を JSON ログとして保存 ----
    option = {
        "k_opt": int(k_opt),    # 最適クラスタ数
        "device": device,
        "random_state": seed,
        "chunk": int(chunk),
        "k_max": int(k_max),
    }
    with OPTION_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(option, f, ensure_ascii=False, indent=2)

    print("✅ ALL DONE")


if __name__ == "__main__":
    main()