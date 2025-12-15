import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

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
from src.evaluation.silhouette import (
    plot_silhouette_samples,
    compute_silhouette_report,
    save_report,
    save_json,
    build_diff_report,
)
from src.evaluation.angles import (
    load_centroids,
    angle_matrix,
    plot_angle_heatmap,
    plot_angle_diff_heatmap,
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
    "silhouette_ref_snv_ckm":   IMG_DIR / "silhouette_samples_ref_snv_ckm.png",
    "silhouette_latent_ckm":    IMG_DIR / "silhouette_samples_latent_ckm.png",

    # 角度系 Heatmap / KDE / Scatter / MDS
    "angle_heat_ref":           IMG_DIR / "angle_heatmap_ref.png",
    "angle_heat_latent":        IMG_DIR / "angle_heatmap_latent.png",
    "angle_heat_diff":          IMG_DIR / "angle_heatmap_diff_lat_minus_ref.png",
    "angle_kde_compare":        IMG_DIR / "angle_kde_ref_vs_latent.png",
    "angle_scatter_compare":    IMG_DIR / "angle_scatter_ref_vs_latent.png",
    "mds_ref":                  IMG_DIR / "center_layout_mds_ref.png",
    "mds_latent":               IMG_DIR / "center_layout_mds_latent.png",
}

# --- モデル／レポート関連 ---
CENTROID_PATHS = {
    "ref_snv_matched": RUNS_DIR / "ref_snv_ckm_matched.pt",
    "latent_ckm":      RUNS_DIR / "latent_ckm.pt",
}

REPORT_DIR = RUNS_DIR / "clustering_report"
REPORT_PATHS = {
    # Silhouette レポート
    "silhouette_ref_snv_ckm": REPORT_DIR / "silhouette_report_ref_snv_ckm.json",
    "silhouette_latent_ckm":  REPORT_DIR / "silhouette_report_latent_ckm.json",

    # 差分レポート
    "diff_ref_vs_latent":     REPORT_DIR / "report_diff_ref_vs_latent.json",
}


# =========================================================
# main
# =========================================================
def main() -> None:
    print("========== [1/9] 設定ファイル読込 ==========")
    cfg = load_config()

    # report セクションは dataclass / dict のどちらでも動くようにしておく
    report_cfg = getattr(cfg, "report", None)
    if report_cfg is not None:
        n_boot     = int(getattr(report_cfg, "n_boot", 5000))
        seed       = int(getattr(report_cfg, "seed", 42))
        data_chunk = int(getattr(report_cfg, "data_chunk", 5_000_000))
        boot_chunk = int(getattr(report_cfg, "boot_chunk", 100))
    else:
        raw = getattr(cfg, "raw", {})  # なければ {} が返る前提
        rep_dict = raw.get("report", {})
        n_boot     = int(rep_dict.get("n_boot", 5000))
        seed       = int(rep_dict.get("seed", 42))
        data_chunk = int(rep_dict.get("data_chunk", 5_000_000))
        boot_chunk = int(rep_dict.get("boot_chunk", 100))

    set_global_seed(seed)
    # 元コード同様 device は固定で "cuda"
    device = "cuda"

    print(f"Loaded config:")
    print(f"  n_boot     = {n_boot}")
    print(f"  seed       = {seed}")
    print(f"  data_chunk = {data_chunk}")
    print(f"  boot_chunk = {boot_chunk}")
    print(f"  device     = {device}")

    # 出力ディレクトリの作成
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --- データ読込 ---
    print("\n========== [2/9] データ読込 ==========")
    X_ref_snv            = np.load(INPUT_PATHS["test_ref_snv"])
    X_labels_ref_snv_ckm = np.load(INPUT_PATHS["labels_ref_snv_ckm"])
    X_latent             = np.load(INPUT_PATHS["test_latent"])
    X_labels_latent_ckm  = np.load(INPUT_PATHS["labels_latent_ckm"])

    print(f"Loaded arrays:")
    print(f"  X_ref_snv: {X_ref_snv.shape}")
    print(f"  X_latent : {X_latent.shape}")
    print(f"  labels_ref_snv_ckm: {np.unique(X_labels_ref_snv_ckm).size} clusters")
    print(f"  labels_latent_ckm : {np.unique(X_labels_latent_ckm).size} clusters")

    # --- Silhouette 計算 ---
    print("\n========== [3/9] Silhouette サンプル計算 (GPU) ==========")
    print("→ baseline (reflectance_snv)")
    silhouette_ref_snv_ckm = silhouette_samples_cosine_gpu(
        X_ref_snv,
        X_labels_ref_snv_ckm,
        device=device,
        chunk=data_chunk,
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
        chunk=data_chunk,
        return_numpy=True,
    )
    print(
        f"  Done. shape={silhouette_latent_ckm.shape}, "
        f"mean={silhouette_latent_ckm.mean():.3f}"
    )

    # --- 可視化（Silhouette） ---
    print("\n========== [4/9] Silhouette 可視化 ==========")
    plot_silhouette_samples(
        silhouette_ref_snv_ckm,
        X_labels_ref_snv_ckm,
        save_path=IMG_PATHS["silhouette_ref_snv_ckm"],
        seed=seed
    )
    print(f"  Saved: {IMG_PATHS['silhouette_ref_snv_ckm']}")

    plot_silhouette_samples(
        silhouette_latent_ckm,
        X_labels_latent_ckm,
        save_path=IMG_PATHS["silhouette_latent_ckm"],
        seed=seed
    )
    print(f"  Saved: {IMG_PATHS['silhouette_latent_ckm']}")

    # --- 詳細レポート ---
    print("\n========== [5/9] 詳細レポート生成 ==========")
    report_ref_snv_ckm = compute_silhouette_report(
        silhouette_ref_snv_ckm,
        X_labels_ref_snv_ckm,
        n_boot=n_boot,
        seed=seed,
        device=device,
        data_chunk=data_chunk,
        boot_chunk=boot_chunk,
    )
    save_report(report_ref_snv_ckm, REPORT_PATHS["silhouette_ref_snv_ckm"])

    report_latent_ckm = compute_silhouette_report(
        silhouette_latent_ckm,
        X_labels_latent_ckm,
        n_boot=n_boot,
        seed=seed,
        device=device,
        data_chunk=data_chunk,
        boot_chunk=boot_chunk,
    )
    save_report(report_latent_ckm, REPORT_PATHS["silhouette_latent_ckm"])

    # --- 差分レポート ---
    print("\n========== [6/9] 差分レポート生成 ==========")
    diff = build_diff_report(report_ref_snv_ckm, report_latent_ckm)
    save_json(diff, REPORT_PATHS["diff_ref_vs_latent"])

    # --- 幾何的検証（中心角度） ---
    print("\n========== [7/9] 角度マージン検証：行列生成 ==========")
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
    print("\n========== [8/9] 角度マージン検証：可視化 ==========")
    plot_angle_heatmap(Theta_ref,   IMG_PATHS["angle_heat_ref"])
    plot_angle_heatmap(Theta_lat,   IMG_PATHS["angle_heat_latent"])
    plot_angle_diff_heatmap(
        Theta_ref,
        Theta_lat,
        IMG_PATHS["angle_heat_diff"],
    )
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

    print("\n========== [9/9] ✅ ALL DONE ==========")


if __name__ == "__main__":
    main()