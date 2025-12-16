from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import colorcet as cc
from sklearn.manifold import MDS


# =========================================================
# 角度行列 & 可視化
# =========================================================
def load_centroids(path: str | Path) -> np.ndarray:
    """
    保存済みクラスタ中心（centroids）をロードして L2 正規化して返す。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"not found: {p}")
    obj = torch.load(p, map_location="cpu")
    C = obj["centroids"] if isinstance(obj, dict) and "centroids" in obj else obj
    if isinstance(C, torch.Tensor):
        C = C.detach().cpu().numpy()
    C = np.asarray(C, dtype=np.float32)
    if C.ndim != 2:
        raise ValueError(f"centroids must be 2D, got {C.shape}")
    n = np.linalg.norm(C, axis=1, keepdims=True)
    C = C / np.clip(n, 1e-12, None)
    return C


def angle_matrix(C: np.ndarray) -> np.ndarray:
    """
    クラスタ中心行列 C (K, D) から角度行列 Θ (K, K) を計算する。
    """
    C = np.asarray(C, dtype=np.float32)
    if C.ndim != 2:
        raise ValueError("C は 2 次元配列である必要があります。")
    S = C @ C.T
    S = np.clip(S, -1.0, 1.0)
    return np.arccos(S)


def plot_angle_kde_comparison(
    theta_ref_rad: np.ndarray,
    theta_lat_rad: np.ndarray,
    save_path: str | Path
) -> None:
    """
    クラスタ間角度（上三角成分）の分布を KDE で推定し、ref と latent を比較プロット。
    """
    from scipy.stats import gaussian_kde

    tri_ref = theta_ref_rad[np.triu_indices_from(theta_ref_rad, 1)]
    tri_lat = theta_lat_rad[np.triu_indices_from(theta_lat_rad, 1)]
    ref_deg, lat_deg = np.degrees(tri_ref), np.degrees(tri_lat)

    x = np.linspace(0, 180, 1000)
    kde_ref = gaussian_kde(ref_deg, bw_method=0.1)
    kde_lat = gaussian_kde(lat_deg, bw_method=0.1)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(x, kde_ref(x), label=f"Ref(SNV) (μ={ref_deg.mean():.3g}°)", color="tab:blue", lw=2)
    ax.plot(x, kde_lat(x), label=f"Latent (μ={lat_deg.mean():.3g}°)", color="tab:orange", lw=2)
    ax.fill_between(x, kde_ref(x), kde_lat(x), color="gray", alpha=0.2, label="Difference region")

    ax.set_xlabel("Inter-cluster angle (°)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=300)
    plt.close(fig)


def plot_angle_scatter_comparison(
    theta_ref_rad: np.ndarray,
    theta_lat_rad: np.ndarray,
    save_path: str | Path,
) -> None:
    """
    Ref と Latent のクラスタ間角度（上三角ペア）を 2D KDE 等高線で比較可視化。
    """
    from scipy.stats import gaussian_kde

    i, j = np.triu_indices_from(theta_ref_rad, 1)
    ref_deg = np.degrees(theta_ref_rad[i, j])
    lat_deg = np.degrees(theta_lat_rad[i, j])

    x_min, x_max = 0.0, 180.0
    y_min, y_max = 0.0, 180.0

    Xs = np.vstack([ref_deg, lat_deg])
    kde = gaussian_kde(Xs, bw_method="scott")

    gridsize = 200
    xs = np.linspace(x_min, x_max, gridsize)
    ys = np.linspace(y_min, y_max, gridsize)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    pts = np.vstack([XX.ravel(), YY.ravel()])
    Z = kde(pts).reshape(XX.shape)
    Z = Z / (Z.max() + 1e-12)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    cf = ax.contourf(XX, YY, Z, levels=20, cmap="viridis", vmin=0, vmax=1)
    ax.contour(XX, YY, Z, levels=10, colors="k", linewidths=0.3, alpha=0.6)
    ax.plot([x_min, x_max], [y_min, y_max], "r--", lw=1)
    ax.set_xlabel("Ref(SNV) angle (°)")
    ax.set_ylabel("Latent angle (°)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label("Relative density (KDE)")
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    plt.tight_layout()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=300)
    plt.close(fig)


def plot_mds_layout_from_angles(
    theta_rad: np.ndarray,
    save_path: str | Path,
    *,
    seed: Optional[int] = 42,
    show_ticks: bool = False,
) -> None:
    """
    角度行列（radian, 形状: K×K）を距離行列とみなし MDS により 2D に射影して可視化。
    """
    D = np.asarray(theta_rad)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("theta_rad は正方の 2 次元配列である必要があります。")
    n_clusters = int(D.shape[0])

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=seed,
        n_init=4,
        max_iter=300,
    )
    coords = mds.fit_transform(D)

    colors = list(cc.glasbey_light[:max(n_clusters, 1)])
    cmap = {i: colors[i % len(colors)] for i in range(n_clusters)}

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    for i, (x, y) in enumerate(coords):
        ax.scatter(
            x,
            y,
            s=250,
            color=cmap[i],
            label=f"{i}",
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        ax.text(
            x,
            y,
            str(i),
            fontsize=10,
            ha="center",
            va="center",
            color="black",
            fontweight="bold",
        )

    ax.set_aspect("equal", adjustable="datalim")
    if show_ticks:
        ax.set_xlabel("MDS-1")
        ax.set_ylabel("MDS-2")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    ax.text(
        0.02,
        0.98,
        f"stress={mds.stress_:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    plt.tight_layout()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=300)
    plt.close(fig)
