from __future__ import annotations

import os
import json
from typing import List, Tuple, Optional

import spectral.io.envi as envi
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt
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


def load(
    folder: str,
    sample_name: str,
    *,
    dim_start: int = 0,
    dim_end: int = 210,  # exclusive
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    指定フォルダ内からサンプルのスペクトルデータを読み込み、
    強度（I）、反射率スペクトル (R)、対応する波長ベクトル (wave) を返す。

    Notes
    -----
    - スペクトル次元は index ベースで [dim_start, dim_end) にクリップされる
    - デフォルトでは 256 → 224 dim（長波長側をカット）
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

    # -------------------------
    # safety check (before clip)
    # -------------------------
    n_dim = I.shape[-1]
    assert 0 <= dim_start < dim_end <= n_dim, (
        f"Invalid clip range: [{dim_start}, {dim_end}) for n_dim={n_dim}"
    )

    # -------------------------
    # reflectance computation
    # -------------------------
    epsilon = 1e-6
    numerator   = (I - D)
    denominator = np.clip(W - D, epsilon, np.inf)
    R = numerator / denominator

    assert np.all(np.isfinite(R)), "Output R contains NaN or Inf."

    # -------------------------
    # dimension clip (index-based)
    # -------------------------
    I = I[..., dim_start:dim_end]
    R = R[..., dim_start:dim_end]
    wave = wave[dim_start:dim_end]

    return I, R, wave


def load_sample_list(sample_name_path: str):
    """
    サンプル名リスト（JSON形式）を読み込む。
    """
    with open(sample_name_path, "r", encoding="utf-8") as f:
        sample_name_list = json.load(f)
    return sample_name_list


def return_binary_data(
    data: np.ndarray,
    *,
    margin_px: int = 3,
):
    """
    強度（I）の L2 ノルム + Otsu で木材領域を粗く推定し、
    そのマスクの「境界から margin_px 以上内側」のみを残す。

    形態学（opening / closing / erosion）は一切使用しない。
    境界ピクセルのスペクトル混合を除外するための
    距離マージンベースのマスク生成。

    Parameters
    ----------
    data : np.ndarray
        強度 I (H, W, C)
    margin_px : int
        境界から何ピクセル内側を採用するか（縁汚染を捨てる幅）

    Returns
    -------
    norm_map : np.ndarray
        L2 ノルムマップ (H, W)
    binary_data : np.ndarray
        内側マージン適用後の木材マスク (H, W), dtype=bool
    """
    if margin_px < 0:
        raise ValueError("margin_px must be >= 0")

    # L2 ノルム
    norm_map = np.linalg.norm(data, axis=2)
    norm_map = np.nan_to_num(norm_map, nan=0.0, posinf=0.0, neginf=0.0)

    # Otsu による粗マスク
    thresh = threshold_otsu(norm_map)
    binary_data = norm_map > thresh

    # 距離マージンで縁を削除
    if margin_px > 0:
        dist = distance_transform_edt(binary_data)
        binary_data = dist >= margin_px

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
    cbar_fontsize: int = 20,
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
        im = main_ax.imshow(img[:, ::-1], cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        im = main_ax.imshow(img[:, ::-1])

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
    各サンプルについて **反射率データ R** と **木材領域マスク mask** を生成して返す。

    処理の流れ
    ----------
    1. `load()` を用いて各サンプルから
       - 強度データ I (H, W, C)
       - 反射率データ R (H, W, C)
       - 波長ベクトル wave
       を読み込む。
    2. 返り値としては **反射率データ R のみ**を `data_list` に格納する。
    3. 木材領域マスクの生成および可視化に関しては、
       **反射率 R ではなく、強度データ I を用いて行う**。
       具体的には：
         - I に対して L2 ノルムマップを計算
         - ノルムマップに Otsu の手法を適用して二値化
    4. 以下のファイルを保存する：
         - ノルムマップ画像（png）
         - 二値マスク画像（png）
         - 二値マスク配列（npy）
         - 波長ベクトル（npy, 初回のみ）
    5. 返り値は **反射率データのリスト** と **二値マスクのリスト**

    Notes
    -----
    - 二値化およびノルム画像作成に「強度 I」を用いるのは、
      センサ由来のラインアーティファクトや
      反射率正規化 (R=(I-D)/(W-D)) による不安定性が
      マスク生成に悪影響を与えるのを避けるためである。
    - 反射率 R は以降のスペクトル解析・学習用途のために
      そのまま保持される。

    Parameters
    ----------
    data_folder : str
        データの格納フォルダ名（train / val / test 等）。
    sample_name_list : List[str]
        サンプル名のリスト。
    width_inch : float, optional
        保存画像の横幅（インチ指定）。

    Returns
    -------
    data_list : List[np.ndarray]
        各サンプルの反射率データ R (H, W, C) のリスト。
    masks : List[np.ndarray]
        各サンプルの木材領域二値マスク (H, W) のリスト。
    """
    data_list: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    wave = None
    intensity_list: List[np.ndarray] = []

    # ----------------------------
    # データ読み込み
    # ----------------------------
    for name in sample_name_list:
        I, R, wave = load(folder=data_folder, sample_name=name)

        assert isinstance(R, np.ndarray) and R.ndim == 3, f"Invalid shape for data {name}"
        assert isinstance(I, np.ndarray) and I.ndim == 3, f"Invalid shape for intensity {name}"

        data_list.append(R)
        intensity_list.append(I)

    # 波長保存（1回だけ）
    if wave is not None:
        SPECTRA_DIR.mkdir(parents=True, exist_ok=True)
        np.save(SPECTRA_DIR / "wavenumber.npy", wave)

    # ----------------------------
    # マスク生成・保存
    # ----------------------------
    for name, I in zip(sample_name_list, intensity_list):
        # NOTE: マスク生成は「強度 I」に対して行う
        norm_map, label = return_binary_data(I)

        _save_tight_image(
            norm_map,
            out_path=NORM_MAP_DIR / f"{name}_I_norm_map.png",
            cmap="jet",
            vmin=100000,
            vmax=400000,
            width_inch=width_inch,
            add_cbar=False,
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
        
        # 反射率のノルム画像も可視化したい. ただし二値化で0だった部分のRは0に置換すること

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