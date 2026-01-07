from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from chemomae.models import ChemoMAE
from chemomae.training import ExtractorConfig, Extractor
from chemomae.utils import set_global_seed

from src.core import (
    load_config,
    LATENT_DIR,
    RUNS_DIR,
    get_reflectance_path,
    get_latent_path,
)


# =========================================================
# main
# =========================================================
def main() -> None:
    # 出力ディレクトリ
    LATENT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- config 読み込み ----
    cfg = load_config()
    tr = cfg.training   # TrainingConfig dataclass
    te = cfg.test       # TestConfig dataclass

    # ---- device / seed ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(tr.seed)

    # ---- training パラメータ ----
    batch_size      = tr.batch_size
    num_workers     = tr.num_workers
    pin_memory      = tr.pin_memory
    persistent_work = tr.persistent_workers

    # --- データ読み込み (SNV 前処理済み配列) ---
    train_snv_path = get_reflectance_path("train", snv=True, downsampled=False)
    val_snv_path   = get_reflectance_path("val",   snv=True, downsampled=False)

    train = np.load(train_snv_path).astype(np.float32, copy=False)
    val   = np.load(val_snv_path).astype(np.float32, copy=False)

    train_ds = TensorDataset(torch.as_tensor(train, dtype=torch.float32))
    val_ds   = TensorDataset(torch.as_tensor(val,   dtype=torch.float32))

    # --- DataLoader の作成 ---
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_work if num_workers > 0 else False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_work if num_workers > 0 else False,
        drop_last=False,
    )

    # ---- 学習済みモデルの読み込み ----
    model = ChemoMAE(**asdict(cfg.model))

    default_weights_path = RUNS_DIR / "best_model.pt"
    weights_cfg = te.weights_path or str(default_weights_path)
    weights_path = Path(weights_cfg)

    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)

    # ---- 潜在表現抽出 (全可視) ----
    extractor_cfg = ExtractorConfig(device=device, amp=False, return_numpy=True)
    extractor = Extractor(model, extractor_cfg)

    Z_train = extractor(train_loader)
    Z_val   = extractor(val_loader)

    # ---- 潜在表現を npy 保存（float32で統一） ----
    latent_train_out = get_latent_path("train")
    latent_val_out   = get_latent_path("val")

    np.save(latent_train_out, np.asarray(Z_train, dtype=np.float32))
    np.save(latent_val_out,   np.asarray(Z_val,   dtype=np.float32))
    
    print("✅ ALL DONE")


if __name__ == "__main__":
    main()