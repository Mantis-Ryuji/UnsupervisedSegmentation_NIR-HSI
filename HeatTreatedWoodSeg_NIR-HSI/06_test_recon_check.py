import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

from dataclasses import asdict
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from chemomae.utils.seed import set_global_seed
from chemomae.preprocessing import cosine_fps_downsample
from chemomae.models import ChemoMAE

from src.core.config import load_config
from src.core.paths import (
    RUNS_DIR,
    IMAGES_DIR,
    get_reflectance_path,
)
from src.preprocessing import trim_rim_by_knn_cosine_gpu
from src.viz.recon import plot_recon_grid


# =========================================================
# パス定義
# =========================================================
TEST_SNV_PATH = get_reflectance_path("test", snv=True, downsampled=False)

SPECTRA_IMG_DIR = IMAGES_DIR / "spectra"
RECON_FIG_PATH = SPECTRA_IMG_DIR / "recon_spectra.png"


# =========================================================
# main
# =========================================================
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- config 読み込み ---
    cfg = load_config()
    tr_cfg = cfg.training
    te_cfg = cfg.test
    
    set_global_seed(tr_cfg.seed)

    batch_size = tr_cfg.batch_size
    num_workers = tr_cfg.num_workers
    pin_memory = tr_cfg.pin_memory
    persistent_work = tr_cfg.persistent_workers

    # --- 出力ディレクトリ作成 ---
    SPECTRA_IMG_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # --- テストデータ読み込み (SNV済み) ---
    test_snv = np.load(TEST_SNV_PATH).astype(np.float32, copy=False)

    # FPS で downsample
    test_down = cosine_fps_downsample(
        test_snv,
        ratio=1e-2,
        seed=tr_cfg.seed,
    )
    # 外れ値・縁トリム
    test_down_trim = trim_rim_by_knn_cosine_gpu(test_down)

    test_dataset = TensorDataset(torch.as_tensor(test_down_trim, dtype=torch.float32))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_work if num_workers > 0 else False,
        drop_last=False,
    )

    # --- 学習済みモデルの読み込み ---
    model = ChemoMAE(**asdict(cfg.model))

    weights_path_cfg = te_cfg.weights_path or str(RUNS_DIR / "best_model.pt")
    weights_path = Path(weights_path_cfg)
    state = torch.load(weights_path, map_location="cpu")

    model.load_state_dict(state, strict=True)
    model.to(device)

    # --- 1バッチだけ再構成して可視化 ---
    x_origin_list = []
    x_recon_list = []
    visible_mask_list = []

    with torch.inference_mode():
        for batch in test_loader:
            x = batch[0].to(device, non_blocking=True)
            x_recon, z, visible_mask = model(x)
            x_origin_list.append(x)
            x_recon_list.append(x_recon)
            visible_mask_list.append(visible_mask)
            break  # 1バッチのみ

    plot_recon_grid(
        x_origin_list,
        x_recon_list,
        visible_mask_list,
        n_blocks=cfg.model.n_patches,
    )
    plt.savefig(RECON_FIG_PATH, dpi=300)
    
    print("✅ ALL DONE")


if __name__ == "__main__":
    main()