from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..core.paths import SPECTRA_DIR


def plot_recon_grid(
    x_origin_list,
    x_recon_list,
    visible_mask_list,
    *,
    n_blocks: int = 32,
    wavenumber_path: Path | str = SPECTRA_DIR / "wavenumber.npy",
    max_plots: int = 10,
    seed: int = 42,
    y_min: float = -3,
    y_max: float = 3,
):
    """
    マスク付き再構成スペクトルをグリッド表示する。
    """
    rng = random.Random(seed)

    x_origin = torch.cat(x_origin_list, dim=0).cpu().numpy()
    x_recon = torch.cat(x_recon_list, dim=0).cpu().numpy()
    vmask = torch.cat(visible_mask_list, dim=0).cpu().numpy()

    N, L = x_origin.shape
    n_sel = min(max_plots, N)

    wavenumber = np.load(wavenumber_path).astype(float)
    assert wavenumber.shape[0] == L, "wavenumber length must match sequence length"

    idxs = rng.sample(range(N), n_sel)

    n_rows, n_cols = 5, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 15), sharex=True)
    axes = axes.ravel()

    proxy_mask = Rectangle(
        (0, 0),
        1,
        1,
        facecolor="lightgray",
        alpha=0.5,
        edgecolor="dimgray",
        linewidth=1.0,
        label="Masked block",
    )

    block_size = L // n_blocks
    assert L % n_blocks == 0, "seq_len must be divisible by n_blocks"

    for ax_idx in range(n_rows * n_cols):
        ax = axes[ax_idx]

        i = idxs[ax_idx]
        x0 = x_origin[i]
        xr = x_recon[i]
        vm = vmask[i].astype(bool)

        ax.set_xlim(wavenumber.min(), wavenumber.max())
        ax.set_ylim(y_min, y_max)

        masked_flat = (~vm)
        block_mask = masked_flat.reshape(n_blocks, block_size).any(axis=1)

        for b, is_masked in enumerate(block_mask):
            if not is_masked:
                continue
            i0, i1 = b * block_size, (b + 1) * block_size
            xL = min(wavenumber[i0], wavenumber[i1 - 1])
            xR = max(wavenumber[i0], wavenumber[i1 - 1])

            ax.add_patch(
                Rectangle(
                    (xL, y_min),
                    (xR - xL),
                    (y_max - y_min),
                    facecolor="lightgray",
                    alpha=0.5,
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=0,
                )
            )

            ax.add_patch(
                Rectangle(
                    (xL, y_min),
                    (xR - xL),
                    (y_max - y_min),
                    facecolor="none",
                    edgecolor="dimgray",
                    linewidth=1.0,
                    zorder=1,
                    antialiased=True,
                )
            )
            ax.vlines(
                [xL, xR],
                y_min,
                y_max,
                colors="dimgray",
                linewidth=1.0,
                zorder=2,
            )

        ax.plot(wavenumber, x0, c="k", lw=1.0, label="Original", zorder=3)
        ax.plot(wavenumber, xr, c="r", lw=1.0, label="Reconstructed", zorder=4)

        ax.grid(False)
        ax.legend(
            handles=[ax.lines[0], ax.lines[1], proxy_mask],
            loc="upper right",
        )

        ax.set_ylabel("Reflectance (SNV)")
        ax.set_xlabel("Wavenumber (nm)")

    plt.tight_layout()