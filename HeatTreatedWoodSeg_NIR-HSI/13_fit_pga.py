from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import numpy as np
import torch

from src.core import RUNS_DIR, get_latent_path
from src.htpd import SphericalPGA1D

LATENT_TRAIN_PATH = get_latent_path("train")

# =========================
# save path
# =========================
PGA_PATH = RUNS_DIR / "pga.pt"


def _load_latent(path: Path | str) -> torch.Tensor:
    x = np.load(path, mmap_mode="r")
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return torch.from_numpy(x)


def main() -> None:
    # latent load
    z_train = _load_latent(LATENT_TRAIN_PATH)  # (N, D), float32

    # =========================
    # fit PGA
    # =========================
    pga = SphericalPGA1D()
    pga.fit(z_train)

    # =========================
    # save (RUNS/pga.pt)
    # =========================
    PGA_PATH.parent.mkdir(parents=True, exist_ok=True)
    pga.save(PGA_PATH)

    print(f"[OK] PGA saved to {PGA_PATH}")


if __name__ == "__main__":
    main()