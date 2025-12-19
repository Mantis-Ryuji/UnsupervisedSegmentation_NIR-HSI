from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.core import RUNS_DIR, DATA_DIR, IMAGES_DIR, get_latent_path
from src.preprocessing import load_sample_list
from src.htpd import SphericalPGA1D


# =========================================================
# Config
# =========================================================
SPLITS: list[str] = ["train", "val", "test"]
MODE: str = "latent_pga_htpd"

PGA_PATH: Path = RUNS_DIR / "pga.pt"


# =========================================================
# Core
# =========================================================
@torch.no_grad()
def predict_htpd(
    pga: SphericalPGA1D,
    latent_mmap: np.ndarray,
) -> np.ndarray:
    """
    latent (N,D) -> HTPD (N,) in [0,1]
    """
    z = torch.from_numpy(np.asarray(latent_mmap, dtype=np.float32))
    htpd = pga.transform(z, normalize=True).cpu().numpy().astype(np.float32, copy=False)
    return htpd


def embed_htpd_to_maps(
    *,
    split: str,
    sample_name_list: list[str],
    htpd_1d: np.ndarray,
    out_npy_dir: Path,
    mode: str,
) -> list[np.ndarray]:
    """
    mask==1 の画素に HTPD を埋め戻し、背景は NaN
    """
    results: list[np.ndarray] = []
    offset = 0

    for name in sample_name_list:
        mask = np.load(DATA_DIR / split / "masks" / f"{name}_mask.npy")
        valid = (mask == 1)
        n_valid = int(valid.sum())

        arr = np.full(mask.shape, np.nan, dtype=np.float32)
        arr[valid] = htpd_1d[offset : offset + n_valid]
        offset += n_valid

        np.save(out_npy_dir / f"{name}_{mode}.npy", arr)
        results.append(arr)

    return results


def save_htpd_heatmap(
    img: np.ndarray,
    out_path: str,
    *,
    width_inch: float = 8.0,
) -> None:
    H, W = img.shape[:2]
    aspect = H / W
    figsize = (width_inch, width_inch * aspect)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig = plt.figure(figsize=figsize)

    cbar_fraction = 0.1
    cbar_pad = 0.02
    cbar_height = 0.8
    cbar_ypos = 0.1

    main_ax = fig.add_axes([0, 0, 1 - cbar_fraction - cbar_pad, 1])
    cax = fig.add_axes([1 - cbar_fraction, cbar_ypos, cbar_fraction, cbar_height])

    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad(alpha=0.0)

    im = main_ax.imshow(
        img[:, ::-1],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    main_ax.set_axis_off()

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("HTPD", fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main() -> None:
    pga = SphericalPGA1D.load(PGA_PATH)

    out_img_dir = IMAGES_DIR / "htpd"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        latent_path = get_latent_path(split)
        sample_name_path = DATA_DIR / split / f"{split}_name_list.json"

        out_npy_dir = DATA_DIR / split / "htpd"
        out_npy_dir.mkdir(parents=True, exist_ok=True)

        latent_mmap = np.load(latent_path, mmap_mode="r")

        htpd_1d = predict_htpd(pga, latent_mmap)

        sample_name_list = load_sample_list(sample_name_path)

        maps = embed_htpd_to_maps(
            split=split,
            sample_name_list=sample_name_list,
            htpd_1d=htpd_1d,
            out_npy_dir=out_npy_dir,
            mode=MODE,
        )

        for name, m in zip(sample_name_list, maps):
            save_htpd_heatmap(
                m,
                str(out_img_dir / f"{name}_{MODE}.png"),
            )

if __name__ == "__main__":
    main()