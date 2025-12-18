from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np

from src.preprocessing import load_sample_list
from src.core.paths import DATA_DIR, IMAGES_DIR, get_split_dir
from src.evaluation.spatial_consistency_score import compute_scs, plot_scs_bar


Mode = Literal["ref_snv_ckm_matched", "latent_ckm"]


def _name_list_path(split: str) -> Path:
    return get_split_dir(split) / f"{split}_name_list.json"


def _label_map_path(split: str, sample_name: str, mode: Mode) -> Path:
    return (
        Path(DATA_DIR)
        / split
        / "labels"
        / f"{sample_name}_cluster_labels_{mode}.npy"
    )


def _load_label_map_2d(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"label_map not found: {path}")
    y = np.load(path)
    if y.ndim != 2:
        raise ValueError(f"label_map must be 2D, got shape={y.shape}, path={path}")
    if not np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.int64)
    return y


def main() -> None:
    # -------------------------------------------------
    # scores
    # -------------------------------------------------
    scores: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "ref_snv": {"train": [], "val": [], "test": []},
        "latent": {"train": [], "val": [], "test": []},
    }

    scs_kwargs = dict(
        valid_mask=None,
        ignore_label=-1,
        connectivity=4,
        small_comp_thresh=16,
    )

    for split in ("train", "val", "test"):
        sample_name_list = load_sample_list(str(_name_list_path(split)))

        for name in sample_name_list:
            # ---- ref_snv ----
            p_ref = _label_map_path(split, name, "ref_snv_ckm_matched")
            y_ref = _load_label_map_2d(p_ref)
            r_ref = compute_scs(y_ref, **scs_kwargs)

            scores["ref_snv"][split].append(
                {
                    "sample": name,
                    "scs_intra": float(r_ref.scs_intra),
                    "scs_inter": float(r_ref.scs_inter),
                }
            )

            # ---- latent ----
            p_lat = _label_map_path(split, name, "latent_ckm")
            y_lat = _load_label_map_2d(p_lat)
            r_lat = compute_scs(y_lat, **scs_kwargs)

            scores["latent"][split].append(
                {
                    "sample": name,
                    "scs_intra": float(r_lat.scs_intra),
                    "scs_inter": float(r_lat.scs_inter),
                }
            )

    # -------------------------------------------------
    # プロット
    # -------------------------------------------------
    out_dir = Path(IMAGES_DIR) / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_scs_bar(
        scores=scores,
        metric="scs_intra",
        save_path=out_dir / "scs_intra_bar.png",
        y_min=0.0,
        y_max=1.1,
        ylabel="Spatial Intra-cluster Consistency"
    )

    plot_scs_bar(
        scores=scores,
        metric="scs_inter",
        save_path=out_dir / "scs_inter_bar.png",
        y_min=0.0,
        y_max=0.3,
        ylabel="Spatial Inter-cluster Discontinuity"
    )


if __name__ == "__main__":
    main()