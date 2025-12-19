from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, PillowWriter


# =========================================================
# 1) 静止画：HTPDグラデーションで重ね描き
# =========================================================
def plot_generated_reflectance_snv_gradient(
    *,
    wave_cm: np.ndarray,                      # (L,) wavenumber (cm^-1)
    x_gen_snv: np.ndarray,                 # (T, L) generated reflectance (SNV)
    htpd_grid: np.ndarray,                 # (T,) HTPD values
    save_path: Union[str, Path],
    figsize: Tuple[float, float] = (10, 5),
    ylims: Optional[Tuple[float, float]] = (-3.0, 3.0),
    cmap: str = "viridis",                 # HTPD low -> high
    lw: float = 1.2,
    alpha: float = 0.9,
    xlabel: str = "Wavenumber (cm$^{-1}$)",
    ylabel: str = "Generated Reflectance (SNV)",
) -> Path:
    """
    HTPD に沿って生成された SNV スペクトルを、1 枚の図に
    グラデーション色で重ね描きする（静止画）。
    """
    wave_cm = np.asarray(wave_cm, float).reshape(-1)
    X = np.asarray(x_gen_snv, float)
    h = np.asarray(htpd_grid, float).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"x_gen_snv must be 2D, got shape={X.shape}")
    T, L = X.shape
    if wave_cm.size != L:
        raise ValueError("wave length must match spectral length")
    if h.size != T:
        raise ValueError("htpd_grid length must match number of spectra")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    norm = Normalize(vmin=float(h.min()), vmax=float(h.max()))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    for i in range(T):
        ax.plot(
            wave_cm,
            X[i],
            color=sm.to_rgba(h[i]),
            lw=lw,
            alpha=alpha,
        )
    
    ax.set_xlim(wave_cm.max(), wave_cm.min())
    ax.set_ylim(*ylims)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("HTPD")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


# =========================================================
# 2) アニメーション：HTPD 進行 GIF
# =========================================================
def make_generated_reflectance_snv_gif(
    *,
    wave_cm: np.ndarray,               # (L,)
    x_gen_snv: np.ndarray,          # (T, L)
    htpd_grid: np.ndarray,          # (T,)
    save_path: Union[str, Path],
    fps: int = 20,
    dpi: int = 200,
    figsize: Tuple[float, float] = (10, 5),
    ylims: Optional[Tuple[float, float]] = (-3.0, 3.0),
    xlabel: str = "Wavenumber (cm$^{-1}$)",
    ylabel: str = "Generated Reflectance (SNV)",
) -> Path:
    """
    HTPD に沿って生成スペクトルが変化する様子を GIF にする。
    """
    wave_cm = np.asarray(wave_cm, float).reshape(-1)
    X = np.asarray(x_gen_snv, float)
    h = np.asarray(htpd_grid, float).reshape(-1)

    T, L = X.shape
    if wave_cm.size != L:
        raise ValueError("wave length must match spectral length")
    if h.size != T:
        raise ValueError("htpd_grid length must match number of frames")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    (line,) = ax.plot(wave_cm, X[0], lw=2.0)

    ax.set_xlim(wave_cm.max(), wave_cm.min())
    ax.set_ylim(*ylims)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    title = ax.set_title(f"HTPD={h[0]:.2f}")

    def update(i: int):
        line.set_ydata(X[i])
        title.set_text(f"HTPD={h[i]:.2f}")
        return (line, title)

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=True)
    anim.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return save_path


# =========================================================
# 3) 生成：HTPD → PGA inverse → decoder → SNV
# =========================================================
def generate_reflectance_snv_from_htpd_grid(
    *,
    model,
    pga,
    htpd_grid: np.ndarray,
    device: str = "cuda",
    batch_size: int = 128,
) -> np.ndarray:
    """
    HTPD グリッドから
      PGA inverse → latent z → decoder
    を経て Generated Reflectance (SNV) を生成する。

    Returns
    -------
    X : ndarray, shape (T, L)
        生成された SNV スペクトル
    """

    h = np.asarray(htpd_grid, dtype=float).reshape(-1)  # (T,)
    T = h.size

    # --- PGA inverse: (T, latent_dim)
    z_out = pga.inverse_transform(h, normalized=True)

    # z_out が np.ndarray / torch.Tensor のどちらでもOK
    if isinstance(z_out, torch.Tensor):
        z = z_out.to(device=device, dtype=torch.float32)
    else:
        z = torch.from_numpy(np.asarray(z_out, dtype=np.float32)).to(device)

    outs = []
    with torch.no_grad():
        for i in range(0, T, batch_size):
            zb = z[i : i + batch_size]
            x = model.decoder(zb)  # (B, L) を想定
            outs.append(x.detach().cpu().numpy())

    X = np.concatenate(outs, axis=0)
    return X
