import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

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

IMG_LOG_DIR       = IMAGES_DIR / "logs"
ELBOW_IMG_PATH    = IMG_LOG_DIR / "elbow_k.png"
SIL_IMG_PATH      = IMG_LOG_DIR / "silhouette_k_opt.png"

CENTROID_OUT_PATH = get_centroid_path("latent")         # runs/latent_ckm.pt
REPORT_DIR       = RUNS_DIR / "clustering_report"
OPTION_JSON_PATH  = REPORT_DIR / "option_ckm.json"


# =========================================================
# ユーティリティ
# =========================================================
def load_latent(train_path: Path | str, val_path: Path | str) -> torch.Tensor:
    """
    学習済みChemoMAEで抽出した潜在表現 (train/val) を読み込んで結合する関数。

    - np.load(mmap_mode="r") でメモリ節約しつつ読み込み
    - dtype を float32 に統一して torch.Tensor 化
    """
    lt = np.load(train_path, mmap_mode="r")
    lv = np.load(val_path, mmap_mode="r")
    latent_cat = np.concatenate([lt, lv], axis=0)
    if latent_cat.dtype != np.float32:
        latent_cat = latent_cat.astype(np.float32, copy=False)
    latent = torch.from_numpy(latent_cat)
    return latent


# =========================================================
# main
# =========================================================
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

    # --- 潜在表現の読み込み (train + val を連結) ---
    latent = load_latent(
        LATENT_TRAIN_PATH,
        LATENT_VAL_PATH,
    )

    # --- クラスタ数 K を自動決定（曲率法 / エルボー法）---
    k_list, inertias, k_elbow, idx, kappa = elbow_ckmeans(
        CosineKMeans,   # クラスタリングアルゴリズム（ここではCosineKMeans）
        latent,
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
    # Silhouette による最終 k_opt の決定
    #   - k_elbow ± 5 の範囲（[2, k_max] にクリップ）で
    #     cosine silhouette を最大化する k を採用
    # =====================================================
    k_candidates = [
        k for k in range(k_elbow - 5, k_elbow + 6)
        if 2 <= k <= k_max
    ]

    sil_scores = []
    best_k = None
    best_sil = -1.0

    for k in k_candidates:
        print(f"[Silhouette] k = {k}")

        # --- クラスタリング ---
        ckm_tmp = CosineKMeans(
            n_components=int(k),
            device=device,
            random_state=seed,
        )
        labels = ckm_tmp.fit_predict(latent, chunk=chunk)

        # --- Silhouette スコア算出（GPU）---
        sil = silhouette_score_cosine_gpu(
            latent,
            labels,
            device=device,
            chunk=chunk,
            return_numpy=True,
        )
        sil_scores.append(sil)
        print(f"  silhouette_score = {sil:.6f}")

        if sil > best_sil:
            best_sil = sil
            best_k = k

    k_opt = int(best_k)

    print("\n===== Silhouette Selection Result =====")
    print(f"  k_elbow = {int(k_elbow)}")
    print(f"  k_opt   = {k_opt}  (silhouette = {best_sil:.6f})\n")

    # Silhouette プロット（横軸 = k_candidates）
    plt.figure(figsize=(6,4))
    plt.plot(k_candidates, sil_scores, marker="o", label="Silhouette score")
    # 最適 k_opt
    plt.axvline(k_opt, color="red", linestyle="--", label=f"k_opt={k_opt}")
    plt.xticks(k_candidates)
    plt.xlabel("k")
    plt.ylabel("silhouette score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig(SIL_IMG_PATH, dpi=300)
    plt.close()

    # --- CosineKMeans による実クラスタリング ---
    ckm = CosineKMeans(
        n_components=int(k_opt),
        device=device,
        random_state=seed,
    )
    ckm.fit(latent, chunk=chunk)

    # --- 学習済みクラスタ中心を保存 ---
    ckm.save_centroids(CENTROID_OUT_PATH)

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