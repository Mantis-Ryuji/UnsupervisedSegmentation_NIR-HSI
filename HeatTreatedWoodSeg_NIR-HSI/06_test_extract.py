from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
    # --- config 読み込み ---
    cfg = load_config()
    tr = cfg.training   # TrainingConfig
    te = cfg.test       # TestConfig

    # --- device / seed ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(tr.seed)

    # --- DataLoader 設定 ---
    batch_size      = tr.batch_size
    num_workers     = tr.num_workers
    pin_memory      = tr.pin_memory
    persistent_work = tr.persistent_workers

    # --- 出力ディレクトリ作成 ---
    LATENT_DIR.mkdir(parents=True, exist_ok=True)

    # --- テストデータ読み込み (SNV済み) ---
    test_snv_path = get_reflectance_path("test", snv=True, downsampled=False)
    test_snv = np.load(test_snv_path).astype(np.float32, copy=False)

    test_dataset = TensorDataset(torch.as_tensor(test_snv, dtype=torch.float32))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_work if num_workers > 0 else False,
        drop_last=False,
    )

    # --- 学習済みモデルの読み込み ---
    model = ChemoMAE(**asdict(cfg.model))

    default_weights_path = RUNS_DIR / "best_model.pt"
    weights_cfg = te.weights_path or str(default_weights_path)
    state = torch.load(weights_cfg, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)

    # --- Extractor による潜在表現抽出 ---
    extractor_cfg = ExtractorConfig(device=device, amp=False, return_numpy=True)
    extractor = Extractor(model, extractor_cfg)

    Z_test = extractor(test_loader)

    # --- 保存 ---
    latent_out_path = get_latent_path("test")
    np.save(latent_out_path, np.asarray(Z_test, dtype=np.float32))
    
    print("✅ ALL DONE")


if __name__ == "__main__":
    main()