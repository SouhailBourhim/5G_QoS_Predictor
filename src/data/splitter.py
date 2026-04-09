"""
Temporal splitting of 5G QoS feature datasets.

Splits a 90-day dataset into:
  - Train:      days  1–60
  - Validation: days 61–75
  - Test:       days 76–90

No random shuffling. Strict chronological order enforced with assertions.
Outputs saved to data/splits/{slice_type}_{train|val|test}.parquet
"""

import pandas as pd
from pathlib import Path


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    train_days: int = 60,
    val_days: int = 15,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition a time-ordered DataFrame into train / val / test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a datetime-typed column named `timestamp_col`.
    train_days : int
        Number of days in the training set (default: 60).
    val_days : int
        Number of days in the validation set (default: 15).
    timestamp_col : str
        Name of the timestamp column.

    Returns
    -------
    train, val, test : pd.DataFrame

    Raises
    ------
    AssertionError
        If splits overlap or are mis-ordered.
    """
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    start_date = df[timestamp_col].dt.normalize().min()
    train_end = start_date + pd.Timedelta(days=train_days)
    val_end = train_end + pd.Timedelta(days=val_days)

    train = df[df[timestamp_col] < train_end].copy()
    val = df[(df[timestamp_col] >= train_end) & (df[timestamp_col] < val_end)].copy()
    test = df[df[timestamp_col] >= val_end].copy()

    # ── Integrity assertions ──────────────────────────────────────────────────
    assert len(train) > 0, "Train split is empty"
    assert len(val) > 0, "Validation split is empty"
    assert len(test) > 0, "Test split is empty"

    assert train[timestamp_col].max() < val[timestamp_col].min(), (
        "LEAKAGE: train and val timestamps overlap!"
    )
    assert val[timestamp_col].max() < test[timestamp_col].min(), (
        "LEAKAGE: val and test timestamps overlap!"
    )

    return train, val, test


def split_all_slices(
    processed_dir: str | Path = "data/processed",
    splits_dir: str | Path = "data/splits",
    train_days: int = 60,
    val_days: int = 15,
    timestamp_col: str = "timestamp",
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Read all engineered-feature Parquet files, split each, save to data/splits/.

    Looks for files matching: {processed_dir}/{slice_type}.parquet
    Saves: {splits_dir}/{slice_type}_{train|val|test}.parquet

    Returns a nested dict: {slice_type: {'train': df, 'val': df, 'test': df}}
    """
    processed_dir = Path(processed_dir)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    slice_types = ["eMBB", "URLLC", "mMTC"]

    for stype in slice_types:
        path = processed_dir / f"{stype}.parquet"
        if not path.exists():
            print(f"[SKIP] {path} not found — run the feature pipeline first.")
            continue

        print(f"Splitting {stype}…")
        df = pd.read_parquet(path)

        train, val, test = temporal_split(df, train_days, val_days, timestamp_col)

        train.to_parquet(splits_dir / f"{stype}_train.parquet", index=False)
        val.to_parquet(splits_dir / f"{stype}_val.parquet", index=False)
        test.to_parquet(splits_dir / f"{stype}_test.parquet", index=False)

        results[stype] = {"train": train, "val": val, "test": test}

        # ── Summary ──────────────────────────────────────────────────────────
        print(f"  Train : {len(train):>6,} rows  "
              f"{train[timestamp_col].min().date()} → {train[timestamp_col].max().date()}")
        print(f"  Val   : {len(val):>6,} rows  "
              f"{val[timestamp_col].min().date()} → {val[timestamp_col].max().date()}")
        print(f"  Test  : {len(test):>6,} rows  "
              f"{test[timestamp_col].min().date()} → {test[timestamp_col].max().date()}")

        if "violation_in_30min" in df.columns:
            for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
                rate = split_df["violation_in_30min"].mean() * 100
                print(f"  violation_in_30min positive rate [{split_name}]: {rate:.2f}%")

    return results


if __name__ == "__main__":
    split_all_slices()
