import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from chemomae.utils.seed import set_global_seed

from src.preprocessing import (
    load_sample_list,
    return_data_list_and_mask_list,
    return_full_dataset_np,
    return_downsampled_dataset_np,
)
from src.core.config import load_config
from src.core.paths import IMAGES_DIR, get_split_dir


# =========================================================
# 設定
# =========================================================
CONFIG_SPLITS = ["train", "val", "test"]


# =========================================================
# main
# =========================================================
def main() -> None:
    
    cfg = load_config()
    seed = int(cfg.training.seed)
    set_global_seed(seed)
    
    # 出力先ディレクトリ作成
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # 各 split を処理
    for split in CONFIG_SPLITS:
        split_dir = get_split_dir(split)

        # --- サンプル名リスト読み込み ---
        name_list_path = split_dir / f"{split}_name_list.json"
        name_list = load_sample_list(sample_name_path=name_list_path)

        # --- データ読み込み・マスク作成 ---
        data_list, masks = return_data_list_and_mask_list(
            data_folder=split,
            sample_name_list=name_list,
        )

        # --- train のみ downsample dataset 生成 ---
        if split == "train":
            return_downsampled_dataset_np(
                data_folder=split,
                data_list=data_list,
                masks=masks,
                seed=seed
            )

        # --- full dataset 生成 ---
        return_full_dataset_np(
            data_folder=split,
            data_list=data_list,
            masks=masks,
        )

        print(f"preprocessing for {split} : done")
    
    print("✅ ALL DONE")


if __name__ == "__main__":
    main()