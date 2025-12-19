from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from chemomae.utils import set_global_seed
from chemomae.clustering import silhouette_samples_cosine_gpu

from src.core import (
    load_config,
    RUNS_DIR,
    IMAGES_DIR,
    get_reflectance_path,
    get_latent_path,
    get_cluster_label_path,
)
from src.evaluation import (
    plot_silhouette_bar,
    load_centroids,
    angle_matrix,
    plot_angle_kde_comparison,
    plot_angle_scatter_comparison,
    plot_mds_layout_from_angles,
)


# =========================================================
# パス定義
# =========================================================

# --- 入力データ／ラベル ---
# --- 入力データ／ラベル ---
# ここでは **silhouette は全 split（train/val/test）で検証**する。
INPUT_PATHS = {
    # reflectance SNV
    "train_ref_snv":      get_reflectance_path("train", snv=True, downsampled=False),
    "val_ref_snv":        get_reflectance_path("val",   snv=True, downsampled=False),
    "test_ref_snv":       get_reflectance_path("test",  snv=True, downsampled=False),

    # ChemoMAE latent
    "train_latent":       get_latent_path("train"),
    "val_latent":         get_latent_path("val"),
    "test_latent":        get_latent_path("test"),

    # baseline: ref_snv_ckm (Hungarianで latent に整合済み)
    "labels_train_ref_snv_ckm": get_cluster_label_path("train", space="ref_snv", matched=True),
    "labels_val_ref_snv_ckm":   get_cluster_label_path("val",   space="ref_snv", matched=True),
    "labels_test_ref_snv_ckm":  get_cluster_label_path("test",  space="ref_snv", matched=True),

    # latent_ckm ラベル
    "labels_train_latent_ckm":  get_cluster_label_path("train", space="latent", matched=False),
    "labels_val_latent_ckm":    get_cluster_label_path("val",   space="latent", matched=False),
    "labels_test_latent_ckm":   get_cluster_label_path("test",  space="latent", matched=False),
}


# --- 画像出力パス（log 配下に集約） ---
IMG_DIR = IMAGES_DIR / "logs"
IMG_PATHS = {
    # Silhouette
    "silhouette":   IMG_DIR / "silhouette_barplot.png",

    # 角度系 Heatmap / KDE / Scatter / MDS
    "angle_kde_compare":        IMG_DIR / "angle_kde_ref_vs_latent.png",
    "angle_scatter_compare":    IMG_DIR / "angle_scatter_ref_vs_latent.png",
    "mds_ref":                  IMG_DIR / "center_layout_mds_ref.png",
    "mds_latent":               IMG_DIR / "center_layout_mds_latent.png",
}

# --- クラスタ中心 ---
CENTROID_PATHS = {
    "ref_snv_matched": RUNS_DIR / "ref_snv_ckm_matched.pt",
    "latent_ckm":      RUNS_DIR / "latent_ckm.pt",
}

# =========================================================
# main
# =========================================================
def main() -> None:
    cfg = load_config()
    tr_cfg = cfg.training
    cl_cfg = cfg.clustering
    seed = tr_cfg.seed
    chunk = cl_cfg.chunk
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    set_global_seed(seed)

    # 出力ディレクトリの作成
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    # --- データ読込（splitごと） ---
    X_ref_snv = {
        "train": np.load(INPUT_PATHS["train_ref_snv"]),
        "val":   np.load(INPUT_PATHS["val_ref_snv"]),
        "test":  np.load(INPUT_PATHS["test_ref_snv"]),
    }
    y_ref = {
        "train": np.load(INPUT_PATHS["labels_train_ref_snv_ckm"]),
        "val":   np.load(INPUT_PATHS["labels_val_ref_snv_ckm"]),
        "test":  np.load(INPUT_PATHS["labels_test_ref_snv_ckm"]),
    }
    X_latent = {
        "train": np.load(INPUT_PATHS["train_latent"]),
        "val":   np.load(INPUT_PATHS["val_latent"]),
        "test":  np.load(INPUT_PATHS["test_latent"]),
    }
    y_lat = {
        "train": np.load(INPUT_PATHS["labels_train_latent_ckm"]),
        "val":   np.load(INPUT_PATHS["labels_val_latent_ckm"]),
        "test":  np.load(INPUT_PATHS["labels_test_latent_ckm"]),
    }

    # --- Silhouette 計算（splitごと, GPU） ---
    print("========== Silhouette サンプル計算 (GPU) : train/val/test ==========")

    sil_ref = {}
    sil_lat = {}
    for sp in ("train", "val", "test"):
        print(f"→ baseline (reflectance_snv) [{sp}]")
        sil_ref[sp] = silhouette_samples_cosine_gpu(
            X_ref_snv[sp],
            y_ref[sp],
            device=device,
            chunk=chunk,
            return_numpy=True,
        )
        print(f"  Done. shape={sil_ref[sp].shape}, mean={sil_ref[sp].mean():.3f}")

        print(f"→ ChemoMAE latent [{sp}]")
        sil_lat[sp] = silhouette_samples_cosine_gpu(
            X_latent[sp],
            y_lat[sp],
            device=device,
            chunk=chunk,
            return_numpy=True,
        )
        print(f"  Done. shape={sil_lat[sp].shape}, mean={sil_lat[sp].mean():.3f}")

    # --- 可視化（Silhouette: ref_snv vs latent, 各splitで3本ずつ） ---
    plot_silhouette_bar(
        ref_scores=sil_ref,
        ref_cluster_ids=y_ref,
        latent_scores=sil_lat,
        latent_cluster_ids=y_lat,
        save_path=IMG_PATHS["silhouette"],
        splits=("train", "val", "test"),
    )


    # --- 幾何的検証（中心角度） ---
    C_ref = load_centroids(CENTROID_PATHS["ref_snv_matched"])
    C_lat = load_centroids(CENTROID_PATHS["latent_ckm"])
    Theta_ref = angle_matrix(C_ref)
    Theta_lat = angle_matrix(C_lat)

    tri_ref = Theta_ref[np.triu_indices_from(Theta_ref, 1)]
    tri_lat = Theta_lat[np.triu_indices_from(Theta_lat, 1)]
    print(
        f"⟨angle⟩ ref={np.mean(tri_ref):.3f} / "
        f"latent={np.mean(tri_lat):.3f} (rad)"
    )

    # --- 幾何的検証（可視化） ---
    plot_angle_kde_comparison(
        Theta_ref,
        Theta_lat,
        IMG_PATHS["angle_kde_compare"],
    )
    plot_angle_scatter_comparison(
        Theta_ref,
        Theta_lat,
        IMG_PATHS["angle_scatter_compare"],
    )
    plot_mds_layout_from_angles(Theta_ref, IMG_PATHS["mds_ref"], seed=seed)
    plot_mds_layout_from_angles(Theta_lat, IMG_PATHS["mds_latent"], seed=seed)

    print("✅ ALL DONE")


if __name__ == "__main__":
    main()