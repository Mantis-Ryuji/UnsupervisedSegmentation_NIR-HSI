from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def plot_training_history(training_history, save_path=None, show=False):
    # save_path は str でも Path でも受け取れるようにする
    if save_path is not None:
        save_path = Path(save_path)

    # --- 末尾の test レコードを抽出 ---
    last = training_history[-1]
    test_loss = float(last.get("test_loss")) if "test_loss" in last else None

    # --- 学習履歴レコード抽出 ---
    recs = [
        r for r in training_history
        if isinstance(r, dict) and all(k in r for k in ("epoch", "train_loss", "val_loss", "lr"))
    ]
    recs.sort(key=lambda r: r["epoch"])

    epochs = np.array([r["epoch"] for r in recs], dtype=int)
    tr = np.array([r["train_loss"] for r in recs], dtype=float)
    va = np.array([r["val_loss"]   for r in recs], dtype=float)
    lr = np.array([r["lr"]         for r in recs], dtype=float)

    # --- best epoch by val(min) ---
    best_idx = int(np.argmin(va))
    best_epoch = int(epochs[best_idx])
    best_train = float(tr[best_idx])
    best_val   = float(va[best_idx])

    # --- 目盛り（50の倍数）用の範囲を決定 ---
    step = 10
    xmin = int(np.floor(epochs.min() / step) * step)
    xmax = int(np.ceil (epochs.max() / step) * step)

    # --- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)

    # --- (Left) losses ---
    ax = axes[0]
    ax.plot(epochs, tr, label="train_loss")
    ax.plot(epochs, va, label="val_loss")
    ax.scatter([best_epoch], [best_train], s=60,
               label=f"train@best(val) = {best_train:.3g} (ep {best_epoch})")
    ax.scatter([best_epoch], [best_val],   s=60,
               label=f"val@best(val) = {best_val:.3g} (ep {best_epoch})")
    if test_loss is not None:
        ax.scatter([best_epoch], [float(test_loss)], s=60, marker="x",
                   label=f"test@best(val) = {float(test_loss):.3g} (ep {best_epoch})")

    ax.set_xlabel("epoch")
    ax.set_ylabel("Masked MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- (Right) learning rate ---
    ax = axes[1]
    ax.plot(epochs, lr, label="learning rate")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- 両サブプロットの x 軸を 50 の倍数目盛りに統一 ---
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.xaxis.set_major_locator(MultipleLocator(step))

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)