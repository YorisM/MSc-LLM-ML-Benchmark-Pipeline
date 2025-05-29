#!/usr/bin/env python3
"""
Multi-filter extractor for challenges/FOURTOPS/misc/X_dataset_CNN.csv

Outputs:
  • Prints rows for the old rule (10 - all non-zero)
  • Prints rows for the new rule (12 - ≥5 non-zero)
  • Prints & saves the 10 rows with the fewest trailing zeros
"""

from pathlib import Path
import pandas as pd
import numpy as np

CSV_PATH = Path("challenges/FOURTOPS/misc/X_dataset_CNN.csv")
OUTPUT_CSV = Path("top10_fewest_trailing_zeros.csv")


def rows_last10_all_nonzero(df: pd.DataFrame) -> list[list[float]]:
    slice_ = df.iloc[:, -10:]
    return df[(slice_ != 0).all(axis=1)].values.tolist()


def rows_last12_atleast5_nonzero(df: pd.DataFrame) -> list[list[float]]:
    slice_ = df.iloc[:, -12:]
    return df[(slice_ != 0).sum(axis=1) >= 5].values.tolist()


def trailing_zero_count(arr: np.ndarray) -> int:
    """Count consecutive zeros from the end of a 1-D array."""
    return int(np.argmax(arr[::-1] != 0)) if (arr != 0).any() else len(arr)


def top10_fewest_trailing_zeros(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.apply(lambda row: trailing_zero_count(row.values), axis=1)
    top10_idx = counts.nsmallest(10).index
    return df.loc[top10_idx]


def main():
    df = pd.read_csv(CSV_PATH)

    # ---------- Rule 1 ----------
    old_rows = rows_last10_all_nonzero(df)
    print("=== OLD RULE: last 10 columns all non-zero ===")
    for row in old_rows:
        print(row)
    print(f"Total (old rule): {len(old_rows)}\n")

    # ---------- Rule 2 ----------
    new_rows = rows_last12_atleast5_nonzero(df)
    print("=== NEW RULE: last 12 columns contain ≥5 non-zeros ===")
    for row in new_rows:
        print(row)
    print(f"Total (new rule): {len(new_rows)}\n")

    # ---------- Rule 3 ----------
    top10_df = top10_fewest_trailing_zeros(df)
    top10_rows = top10_df.values.tolist()

    print("=== TOP-10 FEWEST TRAILING ZEROS (entire 105-length row) ===")
    for row in top10_rows:
        print(row)
    print("\nSaved to", OUTPUT_CSV.resolve())

    top10_df.to_csv(OUTPUT_CSV, index=False)

    # Return all three result sets if you want them programmatically
    return old_rows, new_rows, top10_rows


if __name__ == "__main__":
    old_set, new_set, top10_set = main()
