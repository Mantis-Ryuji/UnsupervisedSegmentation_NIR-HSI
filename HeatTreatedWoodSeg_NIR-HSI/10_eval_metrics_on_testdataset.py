import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from chemomae.utils.seed import set_global_seed
from chemomae.clustering import silhouette_samples_cosine_gpu

from src.core.config import load_config
from src.core.paths import (
    RUNS_DIR,
    IMAGES_DIR,
    get_reflectance_path,
    get_latent_path,
    get_cluster_label_path,
)
from src.evaluation.silhouette import plot_silhouette_bar
from src.evaluation.angles import (
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
INPUT_PATHS = {
    # reflectance SNV (test)
    "test_ref_snv":       get_reflectance_path("test", snv=True, downsampled=False),
    # ChemoMAE latent (test)
    "test_latent":        get_latent_path("test"),
    # baseline: ref_snv_ckm (Hungarianで latent に整合済み)
    "labels_ref_snv_ckm": get_cluster_label_path("test", space="ref_snv", matched=True),
    # latent_ckm ラベル
    "labels_latent_ckm":  get_cluster_label_path("test", space="latent", matched=False),
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
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # --- データ読込 ---
    X_ref_snv            = np.load(INPUT_PATHS["test_ref_snv"])
    X_labels_ref_snv_ckm = np.load(INPUT_PATHS["labels_ref_snv_ckm"])
    X_latent             = np.load(INPUT_PATHS["test_latent"])
    X_labels_latent_ckm  = np.load(INPUT_PATHS["labels_latent_ckm"])

    # --- Silhouette 計算 ---
    print("\n========== Silhouette サンプル計算 (GPU) ==========")
    print("→ baseline (reflectance_snv)")
    silhouette_ref_snv_ckm = silhouette_samples_cosine_gpu(
        X_ref_snv,
        X_labels_ref_snv_ckm,
        device=device,
        chunk=chunk,
        return_numpy=True,
    )
    print(
        f"  Done. shape={silhouette_ref_snv_ckm.shape}, "
        f"mean={silhouette_ref_snv_ckm.mean():.3f}"
    )

    print("→ ChemoMAE latent")
    silhouette_latent_ckm = silhouette_samples_cosine_gpu(
        X_latent,
        X_labels_latent_ckm,
        device=device,
        chunk=chunk,
        return_numpy=True,
    )
    print(
        f"  Done. shape={silhouette_latent_ckm.shape}, "
        f"mean={silhouette_latent_ckm.mean():.3f}"
    )

    # --- 可視化（Silhouette） ---
    plot_silhouette_bar(
        ref_scores=silhouette_ref_snv_ckm,
        ref_cluster_ids=X_labels_ref_snv_ckm,
        latent_scores=silhouette_latent_ckm,
        latent_cluster_ids=X_labels_latent_ckm,
        save_path=IMG_PATHS['silhouette']
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