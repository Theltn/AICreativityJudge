"""
Score Dataset Pipeline
=======================
1. Load the cleaned parquet dataset
2. Truncate every story to 500 words
3. Build corpus-level IDF stats for the Novelty dimension
4. Score every story across all 5 dimensions (raw)
5. Normalize all scores to 0–10
6. Compute weighted composite
7. Save the fully scored dataset
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize

# Add project root to path so we can import src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.scoring_rubric import (
    score_story_raw,
    normalize_scores,
    compute_composite,
    WEIGHTS,
)


# ── Helpers ────────────────────────────────────────────────────
def truncate_to_words(text, max_words=500):
    """Keep only the first `max_words` words of a text."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def build_idf(documents, min_df=5):
    """
    Build an IDF dictionary from a list of tokenized documents.
    IDF(t) = log(N / df(t))  where df(t) = number of docs containing term t.
    """
    N = len(documents)
    df_counter = Counter()

    print("  Building document-frequency counts...")
    for i, doc_tokens in enumerate(documents):
        unique_tokens = set(doc_tokens)
        df_counter.update(unique_tokens)

        if (i + 1) % 50000 == 0:
            print(f"    ... processed {i+1}/{N} documents")

    print(f"  Vocabulary size (before min_df filter): {len(df_counter)}")

    idf_dict = {}
    for term, df in df_counter.items():
        if df >= min_df:
            idf_dict[term] = math.log(N / df)

    print(f"  Vocabulary size (after min_df={min_df} filter): {len(idf_dict)}")

    mean_idf = np.mean(list(idf_dict.values())) if idf_dict else 1.0
    print(f"  Corpus mean IDF: {mean_idf:.4f}")

    return idf_dict, mean_idf


# ── Main Pipeline ──────────────────────────────────────────────
def main():
    INPUT_FILE = "data/processed/writing_prompts_full_cleaned.parquet"
    OUTPUT_FILE = "data/processed/writing_prompts_scored.parquet"
    CHECKPOINT_FILE = "data/processed/_scoring_checkpoint.parquet"
    MAX_WORDS = 500
    CHECKPOINT_INTERVAL = 2000  # Save progress every N stories

    print("=" * 60)
    print("  SCORING PIPELINE")
    print("=" * 60)

    # ── 1. Load ────────────────────────────────────────────────
    print(f"\n[1/6] Loading dataset from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"  Loaded {len(df)} stories.")

    # ── 2. Truncate ────────────────────────────────────────────
    print(f"\n[2/6] Truncating stories to {MAX_WORDS} words...")
    df["story_truncated"] = df["story"].apply(lambda s: truncate_to_words(s, MAX_WORDS))

    word_counts = df["story_truncated"].str.split().str.len()
    print(f"  After truncation — mean: {word_counts.mean():.0f}, "
          f"median: {word_counts.median():.0f}, max: {word_counts.max()}")

    # ── 3. Build IDF ───────────────────────────────────────────
    print(f"\n[3/6] Tokenizing corpus for IDF computation...")
    t0 = time.time()
    tokenized_corpus = []
    for i, text in enumerate(df["story_truncated"]):
        tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
        tokenized_corpus.append(tokens)
        if (i + 1) % 50000 == 0:
            print(f"  Tokenized {i+1}/{len(df)}...")

    print(f"  Tokenization took {time.time()-t0:.1f}s")

    print("\n  Computing IDF dictionary...")
    idf_dict, corpus_mean_idf = build_idf(tokenized_corpus)

    # ── 4. Score Every Story ───────────────────────────────────
    print(f"\n[4/6] Scoring {len(df)} stories across 5 dimensions...")

    # Check for existing checkpoint
    start_idx = 0
    results = []
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint_df = pd.read_parquet(CHECKPOINT_FILE)
        start_idx = len(checkpoint_df)
        results = checkpoint_df.to_dict("records")
        print(f"  Resuming from checkpoint at index {start_idx}")

    t0 = time.time()
    for i in range(start_idx, len(df)):
        text = df.iloc[i]["story_truncated"]

        try:
            raw = score_story_raw(text, idf_dict=idf_dict, corpus_mean_idf=corpus_mean_idf)
        except Exception as e:
            # Graceful fallback: zero scores for broken text
            raw = {dim: 0.0 for dim in WEIGHTS}
            print(f"  ⚠ Error scoring story {i}: {e}")

        results.append(raw)

        # Progress reporting
        done = i - start_idx + 1
        if done % 500 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            remaining = (len(df) - i - 1) / rate
            print(f"  [{i+1}/{len(df)}]  {rate:.1f} stories/sec  "
                  f"ETA: {remaining/60:.0f} min")

        # Checkpoint
        if done % CHECKPOINT_INTERVAL == 0:
            ckpt_df = pd.DataFrame(results)
            ckpt_df.to_parquet(CHECKPOINT_FILE)

    raw_scores_df = pd.DataFrame(results)
    print(f"\n  Scoring complete in {(time.time()-t0)/60:.1f} minutes.")

    # ── 5. Normalize to 0–10 ──────────────────────────────────
    print("\n[5/6] Normalizing scores to 0–10 range...")
    norm_df = normalize_scores(raw_scores_df)

    # ── 6. Compute Composite ──────────────────────────────────
    print("[6/6] Computing weighted composite score...")
    norm_df["composite_score"] = compute_composite(norm_df)

    # Merge back into the main dataframe
    for col in list(WEIGHTS.keys()) + ["composite_score"]:
        df[col] = norm_df[col].values

    # Save
    df.to_parquet(OUTPUT_FILE)
    print(f"\n  Saved scored dataset to {OUTPUT_FILE}")
    print(f"  Shape: {df.shape}")

    # Summary stats
    print("\n" + "=" * 60)
    print("  SCORE DISTRIBUTION SUMMARY")
    print("=" * 60)
    for col in list(WEIGHTS.keys()) + ["composite_score"]:
        print(f"  {col:25s}  mean={df[col].mean():.2f}  "
              f"std={df[col].std():.2f}  "
              f"min={df[col].min():.2f}  max={df[col].max():.2f}")

    # Cleanup checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("\n  Checkpoint file cleaned up.")

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
