from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from dataclasses import asdict
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from chemomae.utils import set_global_seed
from chemomae.preprocessing import cosine_fps_downsample
from chemomae.models import ChemoMAE

from src.core import (
    load_config,
    RUNS_DIR,
    IMAGES_DIR,
    get_reflectance_path,
)
from src.viz import plot_recon_grid


# =========================================================
# パス定義
# =========================================================
TEST_SNV_PATH = get_reflectance_path("test", snv=True, downsampled=False)

SPECTRA_IMG_DIR = IMAGES_DIR / "spectra"
RECON_FIG_PATH_TMPL = SPECTRA_IMG_DIR / "recon_spectra_{i}.png"


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
        ratio=1e-3,
        seed=tr_cfg.seed,
    )

    test_dataset = TensorDataset(torch.as_tensor(test_down, dtype=torch.float32))
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
    model.eval()

    # --- 5バッチ再構成して可視化（各バッチ1枚保存） ---
    n_batches_to_plot = 5

    with torch.inference_mode():
        for i, batch in enumerate(itertools.islice(itertools.cycle(test_loader), n_batches_to_plot)):
            x = batch[0].to(device, non_blocking=True)
            x_recon, z, visible_mask = model(x)

            plt.close("all")
            plt.figure()

            plot_recon_grid(
                [x],
                [x_recon],
                [visible_mask],
                n_blocks=cfg.model.n_patches,
                seed=tr_cfg.seed + i,
            )

            out_path = Path(str(RECON_FIG_PATH_TMPL).format(i=i))
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close("all")

            print(f"saved: {out_path}")

    print("✅ ALL DONE")


if __name__ == "__main__":
    main()