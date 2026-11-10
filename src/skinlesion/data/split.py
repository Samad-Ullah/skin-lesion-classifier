"""Leakage-free train/val/test split of HAM10000, grouped by lesion.

Run with:  uv run python -m skinlesion.data.split
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _hold_out(df: pd.DataFrame, n_splits: int, seed: int):
    """Hold out ~1/n_splits of df, keeping lesions together and classes balanced."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep_idx, held_idx = next(sgkf.split(df, df["dx"], groups=df["lesion_id"]))
    return df.iloc[keep_idx], df.iloc[held_idx]


def make_splits(meta: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Return image_id, lesion_id, dx and split (train / val / test), about 70/15/15."""
    rest, test = _hold_out(meta, n_splits=7, seed=seed)  # 1/7 ~ 14% test
    train, val = _hold_out(rest, n_splits=6, seed=seed)  # 1/6 of the rest ~ 14% val
    out = pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]
    )
    cols = ["image_id", "lesion_id", "dx", "split"]
    return out[cols].sort_values("image_id").reset_index(drop=True)


def main(
    meta_path: Path = Path("data/raw/HAM10000_metadata.csv"),
    out_path: Path = Path("data/splits.csv"),
) -> None:
    splits = make_splits(pd.read_csv(meta_path))
    splits.to_csv(out_path, index=False)
    print(splits["split"].value_counts())
    print(pd.crosstab(splits["dx"], splits["split"], normalize="columns").round(3))


if __name__ == "__main__":
    main()
