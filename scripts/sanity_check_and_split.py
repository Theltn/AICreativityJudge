"""
Sanity Check & Split Pipeline
===============================
1. Load scored dataset
2. Print top/bottom stories for manual inspection
3. Check dimension correlations
4. Create Train (70%) / Val (15%) / Test (15%) splits
"""

import pandas as pd
import numpy as np

def main():
    INPUT = "data/processed/writing_prompts_scored.parquet"

    print("=" * 60)
    print("  SANITY CHECK & SPLIT")
    print("=" * 60)

    df = pd.read_parquet(INPUT)
    print(f"\nLoaded {len(df)} scored stories.\n")

    dims = ["lexical_richness", "syntactic_complexity", "novelty", "imagery", "narrative_dynamics"]

    # ── 1. Score Distribution ──────────────────────────────────
    print("─" * 60)
    print("  1. SCORE DISTRIBUTIONS")
    print("─" * 60)
    for col in dims + ["composite_score"]:
        q = df[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"\n  {col}")
        print(f"    10th={q[0.1]:.2f}  25th={q[0.25]:.2f}  "
              f"median={q[0.5]:.2f}  75th={q[0.75]:.2f}  90th={q[0.9]:.2f}")

    # ── 2. Correlation Matrix ──────────────────────────────────
    print("\n" + "─" * 60)
    print("  2. DIMENSION CORRELATIONS")
    print("─" * 60)
    corr = df[dims].corr()
    print("\n" + corr.round(3).to_string())

    # ── 3. Top 5 Highest Scored Stories ────────────────────────
    print("\n" + "─" * 60)
    print("  3. TOP 5 HIGHEST COMPOSITE SCORES")
    print("─" * 60)
    top5 = df.nlargest(5, "composite_score")
    for i, (_, row) in enumerate(top5.iterrows()):
        print(f"\n  #{i+1}  composite={row['composite_score']:.2f}")
        print(f"  Prompt: {row['prompt'][:120]}...")
        print(f"  Story:  {row['story_truncated'][:300]}...")

    # ── 4. Bottom 5 Lowest Scored Stories ─────────────────────
    print("\n" + "─" * 60)
    print("  4. BOTTOM 5 LOWEST COMPOSITE SCORES")
    print("─" * 60)
    bot5 = df.nsmallest(5, "composite_score")
    for i, (_, row) in enumerate(bot5.iterrows()):
        print(f"\n  #{i+1}  composite={row['composite_score']:.2f}")
        print(f"  Prompt: {row['prompt'][:120]}...")
        print(f"  Story:  {row['story_truncated'][:300]}...")

    # ── 5. Create Train / Val / Test Splits ───────────────────
    print("\n" + "─" * 60)
    print("  5. CREATING SPLITS (70/15/15)")
    print("─" * 60)

    # Shuffle with fixed seed for reproducibility
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df_shuffled)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df_shuffled.iloc[:train_end]
    val_df = df_shuffled.iloc[train_end:val_end]
    test_df = df_shuffled.iloc[val_end:]

    print(f"\n  Train: {len(train_df)}  (composite mean={train_df['composite_score'].mean():.2f})")
    print(f"  Val:   {len(val_df)}  (composite mean={val_df['composite_score'].mean():.2f})")
    print(f"  Test:  {len(test_df)}  (composite mean={test_df['composite_score'].mean():.2f})")

    # Verify splits have similar distributions
    print(f"\n  Distribution check (composite std):")
    print(f"    Train: {train_df['composite_score'].std():.3f}")
    print(f"    Val:   {val_df['composite_score'].std():.3f}")
    print(f"    Test:  {test_df['composite_score'].std():.3f}")

    # Save
    train_df.to_parquet("data/processed/train.parquet")
    val_df.to_parquet("data/processed/val.parquet")
    test_df.to_parquet("data/processed/test.parquet")

    print(f"\n  Saved splits to data/processed/[train|val|test].parquet")
    print("\n✅ Sanity check & splits complete!")


if __name__ == "__main__":
    main()
