from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from scipy.signal import savgol_filter

from ..core.paths import SPECTRA_DIR, IMAGES_DIR
from .clusters import get_glasbey_with_white


# 画像出力用ディレクトリ
SPECTRA_IMG_DIR = IMAGES_DIR / "spectra"


def _class_avgs_npz_path() -> Path:
    return SPECTRA_DIR / "class_avgs.npz"


def _class_abs_sg_npz_path() -> Path:
    return SPECTRA_DIR / "class_avg_absorbance_sg_deriv.npz"


def _cluster_spectra_png_path() -> Path:
    return SPECTRA_IMG_DIR / "cluster_spectra.png"


def _cluster_spectra_2nd_png_path() -> Path:
    return SPECTRA_IMG_DIR / "cluster_spectra_2nd_derive.png"


@torch.no_grad()
def streaming_class_means_gpu(
    ref: np.ndarray,
    snv: np.ndarray,
    labels: np.ndarray,
    *,
    chunk_size: int = 10_000_000,
    eps: float = 1e-12,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    行方向のストリーミング（チャンク）処理でクラスごとの平均ベクトルを GPU 上で集計する。
    """
    # --- 入力をCPU上のfloat32へ（ピン留めは任意） ---
    ref = np.asarray(ref, dtype=np.float32, order="C")
    snv = np.asarray(snv, dtype=np.float32, order="C")
    labels = np.asarray(labels)
    N, C = ref.shape
    assert snv.shape == ref.shape and labels.shape[0] == N

    # --- クラスID圧縮（0..K-1） ---
    uniq, inv = np.unique(labels, return_inverse=True)
    K = uniq.shape[0]

    dev = torch.device(device)

    # --- GPU上の累積バッファ ---
    S_ref = torch.zeros((K, C), device=dev, dtype=torch.float32)
    S_snv = torch.zeros((K, C), device=dev, dtype=torch.float32)
    S_abs = torch.zeros((K, C), device=dev, dtype=torch.float32)
    S_asn = torch.zeros((K, C), device=dev, dtype=torch.float32)
    Cnts = torch.zeros((K,), device=dev, dtype=torch.int64)

    inv_t = torch.from_numpy(inv).to(dev, non_blocking=True)

    for s in range(0, N, chunk_size):
        t = min(s + chunk_size, N)
        idx = slice(s, t)
        n = t - s
        if n == 0:
            continue

        # --- ラベル（チャンク） ---
        lab = inv_t[idx]  # (n,)

        # --- ref・snv をGPUへ ---
        ref_chunk = torch.from_numpy(ref[idx]).to(dev, non_blocking=True)  # (n,C)
        snv_chunk = torch.from_numpy(snv[idx]).to(dev, non_blocking=True)  # (n,C)

        # --- 吸光度（チャンク内のみで生成） ---
        ref_chunk = torch.clamp(ref_chunk, min=eps, max=1.0)
        absorb = -torch.log10(ref_chunk)  # (n,C)

        # --- 吸光度SNV（行ごと, ddof=1） ---
        mu = absorb.mean(dim=1, keepdim=True)  # (n,1)
        sd = absorb.std(dim=1, keepdim=True, unbiased=True)  # ddof=1
        sd = sd + eps
        asnv = (absorb - mu) / sd  # (n,C)

        # --- クラス集約（scatter_add / index_add） ---
        idx2 = lab.view(-1, 1).expand(-1, C)  # (n,C)
        S_ref = S_ref.scatter_add(0, idx2, ref_chunk)
        S_snv = S_snv.scatter_add(0, idx2, snv_chunk)
        S_abs = S_abs.scatter_add(0, idx2, absorb)
        S_asn = S_asn.scatter_add(0, idx2, asnv)
        Cnts = Cnts.index_add(0, lab, torch.ones_like(lab, dtype=torch.int64))

        # --- 明示的にテンポラリを消す（VRAMピーク抑制の保険） ---
        del ref_chunk, snv_chunk, absorb, mu, sd, asnv, idx2, lab

    # --- CPUへ戻し、平均化（float32） ---
    ref_sum = S_ref.detach().cpu().numpy().astype(np.float32, copy=False)
    snv_sum = S_snv.detach().cpu().numpy().astype(np.float32, copy=False)
    abs_sum = S_abs.detach().cpu().numpy().astype(np.float32, copy=False)
    asn_sum = S_asn.detach().cpu().numpy().astype(np.float32, copy=False)
    counts = Cnts.detach().cpu().numpy()

    denom = np.maximum(counts, 1).astype(np.float32)[:, None]
    ref_mean = ref_sum / denom
    snv_mean = snv_sum / denom
    absorb_mean = abs_sum / denom
    asnv_mean = asn_sum / denom
    return uniq, ref_mean, snv_mean, absorb_mean, asnv_mean


def plot_spectra(
    wavelength_nm: np.ndarray,
    ref: np.ndarray,
    snv: np.ndarray,
    cluster_labels: np.ndarray,
    figure_size=(24, 10),
    legend_threshold: int = 20,
    eps: float = 1e-12,
):
    """
    クラスタごとの平均スペクトルを 2×2 グリッドで可視化し、数値と図を保存する。
    """
    # --- 0) 形状チェック ---
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    ref = np.asarray(ref, dtype=float)
    snv = np.asarray(snv, dtype=float)
    assert ref.ndim == 2 and snv.ndim == 2, "ref/snv must be (N, C)."
    N, C = ref.shape
    assert wavelength_nm.shape[0] == C, "wavelength length must match spectral length."
    cluster_labels = np.asarray(cluster_labels)
    assert cluster_labels.shape[0] == N, "cluster_labels length must match N."

    # --- 1) nm -> cm^-1（カイザー） ---
    assert np.all(wavelength_nm > 0), "wavelength (nm) must be positive."
    wavenumber_cm = 1e7 / wavelength_nm  # cm^-1
    assert np.all(np.isfinite(wavenumber_cm)), "Invalid wavenumber values."

    # --- 2) クラスタ平均 ---
    uniq, class_ref_mean, class_snv_mean, class_abs_mean, class_abs_snv_mean = streaming_class_means_gpu(
        ref, snv, cluster_labels, chunk_size=10_000_000, eps=eps, device="cuda"
    )

    # --- 4) 保存 ---
    SPECTRA_DIR.mkdir(parents=True, exist_ok=True)

    """
    保存される配列（np.savez）は以下の通り：

        - wavenumber_cm : float32 ndarray, shape = (L,)
            波数軸（cm⁻¹）。描画全体で共通の x 軸。

        - uniq : int ndarray, shape = (K,)
            クラスタ ID のリスト（例：0〜K-1）。  
            平均波形の行とクラスタ番号の対応を保持。

        - ref : float32 ndarray, shape = (K, L)
            各クラスタの **反射率（reflectance）** の平均波形。

        - snv : float32 ndarray, shape = (K, L)
            各クラスタの **SNV（Standard Normal Variate）正規化** 反射率の平均波形。

        - absorb : float32 ndarray, shape = (K, L)
            各クラスタの **吸光度（absorbance = log(1/R) など）** の平均波形。

        - absorb_snv : float32 ndarray, shape = (K, L)
            各クラスタの **吸光度の SNV 正規化**（SNV-absorbance）の平均波形。

    ここで、
        - K = クラスタ数 (latent ckm)
        - L = スペクトルの次元数  
    """
    np.savez(
        _class_avgs_npz_path(),
        wavenumber_cm=wavenumber_cm,
        uniq=uniq,
        ref=class_ref_mean,
        snv=class_snv_mean,
        absorb=class_abs_mean,
        absorb_snv=class_abs_snv_mean,
    )

    # --- 5) 可視化 ---
    SPECTRA_IMG_DIR.mkdir(parents=True, exist_ok=True)
    cmap = get_glasbey_with_white(len(uniq) + 1)
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    ax11, ax12 = axes[0, 0], axes[0, 1]
    ax21, ax22 = axes[1, 0], axes[1, 1]

    x_min, x_max = wavenumber_cm.min(), wavenumber_cm.max()

    # ---- (1) Reflectance ----
    for idx in range(len(uniq)):
        ax11.plot(
            wavenumber_cm,
            class_ref_mean[idx],
            c=cmap.colors[idx + 1],
            lw=1,
            label=f"class{int(uniq[idx])}",
        )
    ax11.set_ylabel("Reflectance")
    ax11.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax11.set_ylim(0, 1)
    ax11.set_xlim(x_max, x_min)
    if len(uniq) <= legend_threshold:
        ax11.legend(loc="lower left", ncol=2)
    ax11.grid(True)

    # ---- (2) Reflectance (SNV) ----
    for idx in range(len(uniq)):
        ax12.plot(
            wavenumber_cm,
            class_snv_mean[idx],
            c=cmap.colors[idx + 1],
            lw=1,
            label=f"class{int(uniq[idx])}",
        )
    ax12.set_ylabel("Reflectance (SNV)")
    ax12.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax12.set_ylim(-3, 3)
    ax12.set_xlim(x_max, x_min)
    if len(uniq) <= legend_threshold:
        ax12.legend(loc="lower left", ncol=2)
    ax12.grid(True)

    # ---- (3) Absorbance ----
    for idx in range(len(uniq)):
        ax21.plot(
            wavenumber_cm,
            class_abs_mean[idx],
            c=cmap.colors[idx + 1],
            lw=1,
            label=f"class{int(uniq[idx])}",
        )
    ax21.set_ylabel("Absorbance (-log$_{10}$ R)")
    ax21.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax21.set_ylim(0, 1)
    ax21.set_xlim(x_max, x_min)
    if len(uniq) <= legend_threshold:
        ax21.legend(loc="upper left", ncol=2)
    ax21.grid(True)

    # ---- (4) Absorbance (SNV) ----
    for idx in range(len(uniq)):
        ax22.plot(
            wavenumber_cm,
            class_abs_snv_mean[idx],
            c=cmap.colors[idx + 1],
            lw=1,
            label=f"class{int(uniq[idx])}",
        )
    ax22.set_ylabel("Absorbance (SNV)")
    ax22.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax22.set_ylim(-3, 3)
    ax22.set_xlim(x_max, x_min)
    if len(uniq) <= legend_threshold:
        ax22.legend(loc="upper left", ncol=2)
    ax22.grid(True)

    plt.tight_layout()
    plt.savefig(_cluster_spectra_png_path(), dpi=300)
    plt.close()


def _validate_savgol_params(window_length: int, polyorder: int) -> Tuple[int, int]:
    wl = int(window_length)
    po = int(polyorder)
    if wl % 2 == 0:
        wl += 1
    if wl <= po:
        wl = po + 1 if (po + 1) % 2 == 1 else po + 2
    return wl, po


def plot_spectra_2nd_derive(
    wavelength_nm: np.ndarray,
    ref: np.ndarray,
    cluster_labels: np.ndarray,
    figure_size=(15, 5),
    window_length=7,
    polyorder=3,
    deriv_order=2,
    eps: float = 1e-12,
):
    """
    吸光度のクラスタ平均に対して Savitzky–Golay (SG) の n 次微分（既定: 2次）を計算・可視化する。
    """
    # --- nm → cm^-1 ---
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    assert np.all(wavelength_nm > 0), "wavelength_nm must be positive."
    wavenumber_cm = 1e7 / wavelength_nm
    assert np.all(np.isfinite(wavenumber_cm)), "Invalid wavenumber values."

    # --- 形状 ---
    ref = np.asarray(ref, dtype=float)
    assert ref.ndim == 2, "ref must be (N, C)."
    N, C = ref.shape
    assert wavelength_nm.shape[0] == C, "wavelength length must match spectral length."
    cluster_labels = np.asarray(cluster_labels)
    assert cluster_labels.shape[0] == N, "cluster_labels length must match N."

    # --- 吸光度のクラス平均だけをGPUストリーミングで取得 ---
    dummy_ref_snv = np.empty_like(ref, dtype=np.float32)  # snvは未使用のためダミー
    uniq, _, _, class_abs_mean, _ = streaming_class_means_gpu(
        ref, dummy_ref_snv, cluster_labels, chunk_size=10_000_000, eps=eps, device="cuda"
    )

    # --- SG n次微分（既定: 2次）---
    wl, po = _validate_savgol_params(window_length, polyorder)
    delta = float(np.median(np.abs(np.diff(wavenumber_cm))))
    if not np.isfinite(delta) or delta <= 0:
        delta = 1.0
    abs_sg = savgol_filter(
        class_abs_mean,
        window_length=wl,
        polyorder=po,
        deriv=deriv_order,
        delta=delta,
        axis=1,
        mode="interp",
    )

    # --- 保存 ---
    SPECTRA_DIR.mkdir(parents=True, exist_ok=True)

    """    
    保存される配列（np.savez）は以下の通り：

        - wavenumber_cm : float32 ndarray, shape = (L,)
            波数軸（cm⁻¹）。描画で使用する共通の x 軸。

        - uniq : int ndarray, shape = (K,)
            使用したクラスタ ID の並び（0〜K-1）。  
            平均波形とクラスタ番号の対応関係を保持。

        - absorb_sg_deriv : float32 ndarray, shape = (K, L)
            各クラスタにおける **SG フィルタ適用後の吸光度微分スペクトル** の平均波形。  
            （例：1次微分、2次微分など）

        - deriv_order : int
            微分の次数（例：1, 2）。

        - window_length : int
            SG フィルタのウィンドウ幅（奇数である必要がある）。

        - polyorder : int
            SG フィルタで使用した多項式近似の次数。

        - delta : float
            SG フィルタの連続点間隔（dx に相当）。

    各記号の意味：
        - K = クラスタ数  
        - L = スペクトル次元数
    """
    np.savez(
        _class_abs_sg_npz_path(),
        wavenumber_cm=wavenumber_cm,
        uniq=uniq,
        absorb_sg_deriv=abs_sg,
        deriv_order=deriv_order,
        window_length=wl,
        polyorder=po,
        delta=delta,
    )

    # --- 可視化 ---
    SPECTRA_IMG_DIR.mkdir(parents=True, exist_ok=True)
    cmap = get_glasbey_with_white(len(uniq) + 1)

    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    for idx in range(len(uniq)):
        ax.plot(
            wavenumber_cm,
            abs_sg[idx],
            c=cmap.colors[idx + 1],
            lw=1,
            label=f"class{int(uniq[idx])}",
        )
    ax.set_ylim(-2e-5, 2e-5)
    ax.set_xlim(10000, 4100)  # 右肩上がり表示
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(1e-5))
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorbance 2nd derive (SG)")
    if len(uniq) <= 20:
        ax.legend(loc="upper left", ncol=2)
    ax.grid(True, which="both")
    ax.grid(True, which="minor", linestyle=":", lw=0.5)

    plt.tight_layout()
    plt.savefig(_cluster_spectra_2nd_png_path(), dpi=300)
    plt.close()