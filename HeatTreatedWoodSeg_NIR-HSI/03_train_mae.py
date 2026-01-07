from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import asdict
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from chemomae.models import ChemoMAE
from chemomae.training import (
    Trainer,
    TrainerConfig,
    build_optimizer,
    build_scheduler,
)
from chemomae.utils import set_global_seed

from src.core import load_config, get_reflectance_path


# =========================================================
# main
# =========================================================
def main() -> None:
    # --- config 読み込み ---
    cfg = load_config()
    tr = cfg.training  # TrainingConfig dataclass

    # --- device / seed ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(tr.seed)

    # --- モデル定義 ---
    model = ChemoMAE(**asdict(cfg.model))

    # --- training パラメータ ---
    batch_size = tr.batch_size
    num_workers = tr.num_workers
    pin_memory = tr.pin_memory
    persistent_work = tr.persistent_workers

    base_lr = tr.base_lr
    weight_decay = tr.weight_decay
    betas = tr.betas
    eps = tr.eps

    warmup_epochs = tr.warmup_epochs
    min_lr_scale = tr.min_lr_scale

    epochs = tr.epochs
    early_stop = tr.early_stop_patience

    # --- データ読み込み (SNV 前処理済み配列) ---
    train_downsampled_path = get_reflectance_path("train", snv=True, downsampled=True)
    val_snv_path = get_reflectance_path("val", snv=True, downsampled=False)

    train_downsampled = np.load(train_downsampled_path).astype(np.float32, copy=False)
    val = np.load(val_snv_path).astype(np.float32, copy=False)

    train_downsampled_ds = TensorDataset(
        torch.as_tensor(train_downsampled, dtype=torch.float32)
    )
    val_ds = TensorDataset(torch.as_tensor(val, dtype=torch.float32))

    # --- DataLoader の作成 ---
    train_loader = DataLoader(
        train_downsampled_ds,
        batch_size=batch_size,
        shuffle=True,
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

    # --- Optimizer & Scheduler ---
    opt = build_optimizer(
        model,
        lr=base_lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
    sched = build_scheduler(
        opt,
        steps_per_epoch=max(1, len(train_loader)),
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        min_lr_scale=min_lr_scale,
    )

    # --- Trainer Config ---
    trainer_cfg = TrainerConfig(
        device=device,
        amp=False,
        loss_type="sse",
        reduction="batch_mean",
        early_stop_patience=early_stop,
    )

    # --- 学習開始 ---
    trainer = Trainer(
        model,
        opt,
        train_loader,
        val_loader,
        scheduler=sched,
        cfg=trainer_cfg,
    )
    _ = trainer.fit(epochs=epochs)
    
    print("✅ ALL DONE")


if __name__ == "__main__":
    main()