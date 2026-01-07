from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

from dataclasses import asdict
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from chemomae.models import ChemoMAE
from chemomae.training import TesterConfig, Tester
from chemomae.utils import set_global_seed

from src.core import load_config, RUNS_DIR, IMAGES_DIR, get_reflectance_path
from src.viz import plot_training_history


# =========================================================
# パス定義
# =========================================================
TEST_SNV_PATH = get_reflectance_path("test", snv=True, downsampled=False)
DEFAULT_WEIGHTS_PATH = RUNS_DIR / "best_model.pt"
TRAINING_HISTORY_PATH = RUNS_DIR / "training_history.json"

IMG_LOG_DIR = IMAGES_DIR / "logs"
HISTORY_FIG_PATH = IMG_LOG_DIR / "training_history.png"


# =========================================================
# main
# =========================================================
def main() -> None:
    # --- config 読み込み ---
    cfg = load_config()
    tr_cfg = cfg.training
    test_cfg = cfg.test

    # --- device / seed ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(tr_cfg.seed)

    # --- DataLoader 設定 ---
    batch_size      = tr_cfg.batch_size
    num_workers     = tr_cfg.num_workers
    pin_memory      = tr_cfg.pin_memory
    persistent_work = tr_cfg.persistent_workers

    # --- 出力ディレクトリ作成 ---
    IMG_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # --- テストデータ読み込み (SNV済み) ---
    test_snv = np.load(TEST_SNV_PATH).astype(np.float32, copy=False)
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
    weights_cfg = test_cfg.weights_path or str(DEFAULT_WEIGHTS_PATH)
    weights_path = Path(weights_cfg)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)

    # --- Tester によるテスト損失の計算 ---
    tester_cfg = TesterConfig(device=device, amp=False, loss_type="sse", reduction="batch_mean")
    tester = Tester(model, tester_cfg)
    test_loss = tester(test_loader)
    print(f"Test Loss : {test_loss:.5f}")

    # --- training_history.json を可視化 ---
    with TRAINING_HISTORY_PATH.open("r", encoding="utf-8") as f:
        training_history = json.load(f)

    plot_training_history(training_history, save_path=HISTORY_FIG_PATH)
    
    print("✅ ALL DONE")


if __name__ == "__main__":
    main()