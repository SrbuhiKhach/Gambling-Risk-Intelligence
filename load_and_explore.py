# ============================================================
# 01_load_and_explore.py — Տվյալների բեռնում և հիմնական ուսումնասիրություն
# ============================================================
# ԵՊՀ | Ռիսկային Խաղացողների Բացահայտում | Փուլ 1
#
# Այս ֆայլում.
#   1. բեռնում ենք players.csv և features.csv
#   2. merge ենք դրանք մեկ DataFrame-ի մեջ
#   3. ստուգում ենք missing values, dtypes, basic stats
#   4. պահում ենք մաքուր ֆայլը processed/ թղթապանակում
# ============================================================

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# config.py-ից ներմուծում ենք բոլոր path-երն ու հաստատունները
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PLAYERS_CSV, FEATURES_CSV, PROCESSED_DIR,
    FEATURE_COLS, TARGET_COL, RISK_ORDER
)


# ── 1. Ֆայլերի բեռնում ───────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Բեռնում է players.csv-ը և features.csv-ը։

    Returns:
        players_df  — դեմոգրաֆիկ տվյալներ (2000 տող, 9 սյունակ)
        features_df — վարքագծային ցուցիչներ (2000 տող, 15 սյունակ)
    """
    print("=" * 60)
    print("ՏՎՅԱԼՆԵՐԻ ԲԵՌՆՈՒՄ")
    print("=" * 60)

    players_df  = pd.read_csv(PLAYERS_CSV)
    features_df = pd.read_csv(FEATURES_CSV)

    print(f"✓ players.csv   → {players_df.shape[0]:,} տող, {players_df.shape[1]} սյունակ")
    print(f"✓ features.csv  → {features_df.shape[0]:,} տող, {features_df.shape[1]} սյունակ")

    return players_df, features_df


# ── 2. Merge ─────────────────────────────────────────────────────────────────

def merge_datasets(players_df: pd.DataFrame,
                   features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Միավորում է players.csv-ը և features.csv-ը player_id բանալիով։

    left join — players-ն է հիմքը, features-ը կցվում է կողքից։
    how='inner' → պահում ենք միայն երկուսում գոյություն ունեցող player_id-ները
    (ստուգված. երկու ֆայլում էլ 2000 նույն player_id-ները)
    """
    print("\n" + "=" * 60)
    print("MERGE — players + features")
    print("=" * 60)

    # features.csv-ում կան risk_level, risk_label կրկնորդ սյունակներ
    # դրանք ջնջում ենք features-ից merge-ից առաջ (players-ից կօգտագործենք)
    features_clean = features_df.drop(
        columns=["risk_level", "risk_label"], errors="ignore"
    )

    merged = pd.merge(
        players_df,
        features_clean,
        on="player_id",
        how="inner"
    )

    print(f"✓ Merged DataFrame → {merged.shape[0]:,} տող, {merged.shape[1]} սյունակ")
    return merged


# ── 3. Տվյալների ստուգում ────────────────────────────────────────────────────

def validate_data(df: pd.DataFrame) -> None:
    """
    Ստուգում է.
      - missing values (NaN)
      - data types
      - risk_label-ի բաշխումը (class imbalance)
      - feature-ների վիճակագրություն
    """
    print("\n" + "=" * 60)
    print("ՏՎՅԱԼՆԵՐԻ ՍՏՈՒԳՈՒՄ")
    print("=" * 60)

    # 3a. Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("✓ Missing values չկա — բոլոր դաշտերը լրիվ են")
    else:
        print("⚠ Missing values:")
        print(missing)

    # 3b. Data types
    print(f"\n{'Սյունակ':<25} {'Տիպ'}")
    print("-" * 40)
    for col, dtype in df.dtypes.items():
        print(f"  {col:<23} {dtype}")

    # 3c. Risk label բաշխում (class imbalance)
    print("\n" + "=" * 60)
    print("ՌԻՍԿԻ ԴԱՍԵՐԻ ԲԱՇԽՈՒՄ")
    print("=" * 60)

    # 4 դաս (ordinal)
    risk_counts = df["risk_level"].value_counts()
    risk_pct    = df["risk_level"].value_counts(normalize=True) * 100

    print(f"\n  {'Դաս':<15} {'Քանակ':>8} {'%':>8}")
    print("  " + "-" * 33)
    for level in RISK_ORDER:
        if level in risk_counts.index:
            print(f"  {level:<15} {risk_counts[level]:>8,} {risk_pct[level]:>7.1f}%")

    # Binary label (0/1)
    pos = df[TARGET_COL].sum()
    neg = len(df) - pos
    print(f"\n  Binary label (risk_label):")
    print(f"  0 (low risk)  → {neg:,} ({neg/len(df)*100:.1f}%)")
    print(f"  1 (high risk) → {pos:,} ({pos/len(df)*100:.1f}%)")
    print(f"\n  ⚠ Class imbalance ratio = {neg/pos:.1f}:1  (կօգտագործենք SMOTE)")

    # 3d. Feature-ների հիմնական վիճակագրություն
    print("\n" + "=" * 60)
    print("FEATURE-ՆԵՐԻ ՎԻՃԱԿԱԳՐՈՒԹՅՈՒՆ (9 ցուցիչ)")
    print("=" * 60)
    stats = df[FEATURE_COLS].describe().round(3)
    print(stats.to_string())


# ── 4. Պահել processed ֆայլը ─────────────────────────────────────────────────

def save_processed(df: pd.DataFrame) -> Path:
    """
    Պահում է merged DataFrame-ը processed/ թղթապանակում։
    Հետագա ֆայլերը (EDA, modeling) կբեռնեն հենց սա։
    """
    out_path = PROCESSED_DIR / "dataset_merged.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Պահված → {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Քայլ 1. Բեռնել
    players_df, features_df = load_data()

    # Քայլ 2. Merge
    df = merge_datasets(players_df, features_df)

    # Քայլ 3. Ստուգել
    validate_data(df)

    # Քայլ 4. Պահել
    save_processed(df)

    print("\n" + "=" * 60)
    print("✅ 01_load_and_explore.py — ԱՎԱՐՏՎԱԾ")
    print("   Հաջorд  →  python 02_eda.py")
    print("=" * 60)

    return df


if __name__ == "__main__":
    df = main()