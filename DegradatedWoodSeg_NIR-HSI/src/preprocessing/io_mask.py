"""
To Do
二値化は強度に対して行うように修正
HeatTreatedWoodSeg_NIR-HSI 参考
"""

from __future__ import annotations

import os
import json
from typing import List, Tuple, Optional

import spectral.io.envi as envi
from tqdm import tqdm
import numpy as np
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

from chemomae.preprocessing import SNVScaler

from ..core.paths import (
    WHITE_DARK_DIR,
    SPECTRA_DIR,
    IMAGES_DIR,
    get_split_dir,
    get_sample_dir,
    get_mask_dir,
)

# 画像出力用ディレクトリ
NORM_MAP_DIR = IMAGES_DIR / "norm_maps"
MASK_IMG_DIR = IMAGES_DIR / "masks"


def load(folder: str, sample_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    指定フォルダ内からサンプルのスペクトルデータを読み込み、
    反射率スペクトル (R) と対応する波長ベクトル (wave) を返す。
    """
    dark_hdr  = WHITE_DARK_DIR / "D.hdr"
    dark_raw  = WHITE_DARK_DIR / "D.raw"
    white_hdr = WHITE_DARK_DIR / "W.hdr"
    white_raw = WHITE_DARK_DIR / "W.raw"

    sample_hdr = get_sample_dir(folder) / f"{sample_name}.hdr"
    sample_raw = get_sample_dir(folder) / f"{sample_name}.raw"

    dark   = envi.open(str(dark_hdr), str(dark_raw))
    white  = envi.open(str(white_hdr), str(white_raw))
    sample = envi.open(str(sample_hdr), str(sample_raw))

    D = np.array(dark.load(), dtype=np.float32)
    W = np.array(white.load(), dtype=np.float32)
    I = np.array(sample.load(), dtype=np.float32)

    wave_raw = sample.metadata.get("wavelength")
    wave = np.array(wave_raw, dtype=np.float32)

    epsilon = 1e-6
    numerator   = (I - D)
    denominator = np.clip(W - D, epsilon, np.inf)
    R = numerator / denominator

    assert np.all(np.isfinite(R)), "Output R contains NaN or Inf."

    return R, wave


def load_sample_list(sample_name_path: str):
    """
    サンプル名リスト（JSON形式）を読み込む。
    """
    with open(sample_name_path, "r", encoding="utf-8") as f:
        sample_name_list = json.load(f)
    return sample_name_list


def return_binary_data(data: np.ndarray):
    """
    L2ノルムマップ + Otsu で木材領域マスクを生成。
    """
    norm_map: np.ndarray = np.linalg.norm(data, axis=2)
    norm_map = np.nan_to_num(norm_map, nan=0.0, posinf=0.0, neginf=0.0)

    thresh = threshold_otsu(norm_map)
    binary_data = norm_map > thresh

    return norm_map, binary_data


def _save_tight_image(
    img: np.ndarray,
    out_path: str,
    *,
    cmap: str = "jet",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    width_inch: float = 8.0,
    add_cbar: bool = True,
    cbar_label: Optional[str] = None,
    cbar_fontsize: int = 30,
    cbar_ticks: Optional[List[float]] = None,
    cbar_fraction: float = 0.1,
    cbar_pad: float = 0.02,
    cbar_height: float = 0.8,
    cbar_ypos: float = 0.1,
):
    """
    画像をカラーバー付きで「余白最小」で保存するユーティリティ関数。
    """
    H, W = img.shape[:2]
    aspect = H / W
    figsize = (width_inch, width_inch * aspect)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig = plt.figure(figsize=figsize)

    if add_cbar:
        main_ax = fig.add_axes([0, 0, 1 - cbar_fraction - cbar_pad, 1])
        cax = fig.add_axes(
            [
                1 - cbar_fraction,
                cbar_ypos,
                cbar_fraction,
                cbar_height,
            ]
        )
    else:
        main_ax = fig.add_axes([0, 0, 1, 1])
        cax = None

    if vmin is None:
        vmin = float(np.nanmin(img))
    if vmax is None:
        vmax = float(np.nanmax(img))

    if img.ndim == 2:
        im = main_ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        im = main_ax.imshow(img)

    main_ax.set_axis_off()

    if add_cbar and cax is not None:
        cbar = fig.colorbar(im, cax=cax)
        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=cbar_fontsize)
        cbar.ax.tick_params(labelsize=cbar_fontsize)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def return_data_list_and_mask_list(
    data_folder: str,
    sample_name_list: List[str],
    width_inch: float = 8.0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    各サンプルについて R と mask を生成し、norm_map/mask画像・mask.npy保存。
    """
    data_list: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    wave = None
    for name in sample_name_list:
        data, wave = load(folder=data_folder, sample_name=name)
        assert isinstance(data, np.ndarray) and data.ndim == 3, f"Invalid shape for data {name}"
        data_list.append(data)

    if wave is not None:
        SPECTRA_DIR.mkdir(parents=True, exist_ok=True)
        np.save(SPECTRA_DIR / "wavenumber.npy", wave)

    for name, data in zip(sample_name_list, data_list):
        norm, label = return_binary_data(data)

        _save_tight_image(
            norm,
            out_path=NORM_MAP_DIR / f"{name}_norm_map.png",
            cmap="jet",
            vmin=0,
            vmax=16,
            width_inch=width_inch,
            add_cbar=True,
        )

        _save_tight_image(
            label,
            out_path=MASK_IMG_DIR / f"{name}_masked_img.png",
            cmap="gray",
            vmin=0,
            vmax=1,
            width_inch=width_inch,
            add_cbar=True,
        )

        mask_dir = get_mask_dir(data_folder)
        mask_dir.mkdir(parents=True, exist_ok=True)
        np.save(mask_dir / f"{name}_mask.npy", label)
        masks.append(label)

    return data_list, masks


def snv(data: np.ndarray) -> np.ndarray:
    """
    画像単位の SNV（各ピクセルのスペクトルを1本ずつ SNV）。
    """
    scaler = SNVScaler()
    H, W, C = data.shape
    data_reshaped = data.reshape(-1, C)
    snv_data = scaler.transform(data_reshaped).reshape(H, W, C)
    return snv_data


def return_full_dataset_np(
    data_folder: str,
    data_list: List[np.ndarray],
    masks: List[np.ndarray],
):
    """
    SNV 前後の木材領域スペクトルを抽出し、
    data/{data_folder}/reflectance.npy / reflectance_snv.npy に保存。
    """
    data_snv_list = [snv(data) for data in data_list]

    wood_ref = []
    wood_ref_snv = []

    for i, (data, data_snv, mask) in enumerate(
        tqdm(zip(data_list, data_snv_list, masks), total=len(data_list), desc="Extracting wood regions")
    ):
        H, W, C = data.shape

        flat_mask = mask.reshape(-1)
        flat_data = data.reshape(-1, C)
        flat_snv = data_snv.reshape(-1, C)

        selected = flat_mask == 1
        if not np.any(selected):
            tqdm.write(f"Warning: No wood pixels found in sample {i}, skipping.")
            continue

        wood_ref.append(flat_data[selected])
        wood_ref_snv.append(flat_snv[selected])

    if not wood_ref:
        raise ValueError("No wood pixels found in any samples. Check masks or data.")

    wood_ref = np.concatenate(wood_ref, axis=0)
    wood_ref_snv = np.concatenate(wood_ref_snv, axis=0)

    out_dir = get_split_dir(data_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "reflectance.npy", wood_ref)
    np.save(out_dir / "reflectance_snv.npy", wood_ref_snv)

    return wood_ref, wood_ref_snv