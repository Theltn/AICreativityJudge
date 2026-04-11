"""
Score LLM Stories & Generate Comparison Plots
===============================================
Reads stories from data/llm_stories/[LLM_NAME]/*.txt,
scores each through the trained RoBERTa model,
and generates comparison visualizations.

Run: python scripts/score_llm_stories.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import RobertaTokenizer, RobertaForSequenceClassification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.scoring_rubric import (
    lexical_richness, syntactic_complexity,
    novelty_score, imagery_score, narrative_dynamics,
)

# ── Config ──────────────────────────────────────────────────
STORIES_DIR = "data/llm_stories"
MODEL_DIR = "data/models/roberta_creativity_model"
STATS_PATH = "data/models/corpus_stats.json"
OUTPUT_DIR = "data/llm_stories/results"
LLMS = ["ChatGPT", "Claude", "Gemini", "Perplexity", "Copilot"]
LLM_COLORS = {
    "ChatGPT": "#10a37f",
    "Claude": "#d4a574",
    "Gemini": "#4285f4",
    "Perplexity": "#20b2aa",
    "Copilot": "#9b59b6",
    "Human (Corpus)": "#ff5555",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Model ──────────────────────────────────────────────
print("Loading RoBERTa model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
model = model.to(device)
model.eval()

with open(STATS_PATH) as f:
    corpus_stats = json.load(f)

print(f"Model loaded on {device}")


def roberta_predict(text):
    """Run RoBERTa inference on a single text."""
    inputs = tokenizer(
        text, max_length=512, padding="max_length",
        truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        score = model(**inputs).logits.squeeze().item()
    return max(0.0, min(10.0, score))


def truncate(text, max_words=500):
    words = text.split()
    return " ".join(words[:max_words])


def score_rubric(text):
    """Score raw rubric dimensions and normalize."""
    lex = max(0, min(10, (lexical_richness(text) - 0.2) / 0.6 * 10))
    syn = max(0, min(10, (syntactic_complexity(text) - 1.0) / 6.0 * 10))
    nov = max(0, min(10, (novelty_score(text) - 0.2) / 0.6 * 10))
    img = max(0, min(10, imagery_score(text) / 0.12 * 10))
    nar = max(0, min(10, narrative_dynamics(text) / 0.4 * 10))
    composite = 0.20*lex + 0.15*syn + 0.25*nov + 0.20*img + 0.20*nar
    return {
        "lexical_richness": round(lex, 2),
        "syntactic_complexity": round(syn, 2),
        "novelty": round(nov, 2),
        "imagery": round(img, 2),
        "narrative_dynamics": round(nar, 2),
        "rubric_composite": round(composite, 2),
    }


# ── Collect & Score Stories ─────────────────────────────────
print("\nScoring stories...")
rows = []
for llm in LLMS:
    llm_dir = os.path.join(STORIES_DIR, llm)
    if not os.path.isdir(llm_dir):
        print(f"  ⚠ No folder for {llm}, skipping")
        continue

    files = sorted([f for f in os.listdir(llm_dir) if f.endswith(".txt")])
    if not files:
        print(f"  ⚠ No .txt files in {llm}/")
        continue

    print(f"  {llm}: {len(files)} stories")
    for fname in files:
        prompt_id = fname.replace(".txt", "")
        text = open(os.path.join(llm_dir, fname), encoding="utf-8").read().strip()
        text = truncate(text, 500)
        word_count = len(text.split())

        # RoBERTa prediction
        rob = roberta_predict(text)

        # Rubric dimensions
        rubric = score_rubric(text)

        rows.append({
            "llm": llm,
            "prompt_id": prompt_id,
            "word_count": word_count,
            "roberta_score": round(rob, 2),
            **rubric,
        })

if not rows:
    print("\n❌ No stories found! See data/llm_stories/README.md for collection instructions.")
    sys.exit(1)

df = pd.DataFrame(rows)
df.to_csv(f"{OUTPUT_DIR}/llm_scores.csv", index=False)
print(f"\nScored {len(df)} stories total. Saved to {OUTPUT_DIR}/llm_scores.csv")

# ── Human Baseline Stats ───────────────────────────────────
human_mean = corpus_stats["composite_score"]["mean"]
human_std = corpus_stats["composite_score"]["std"]


# ══════════════════════════════════════════════════════════════
#  VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a",
    "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "font.family": "monospace",
    "axes.edgecolor": "#333",
    "grid.color": "#222",
})

# 1. Box Plot — RoBERTa Scores by LLM
print("\nGenerating plots...")
fig, ax = plt.subplots(figsize=(12, 7))

llms_with_data = df["llm"].unique().tolist()
box_data = [df[df["llm"] == llm]["roberta_score"].values for llm in llms_with_data]
colors = [LLM_COLORS.get(llm, "#888") for llm in llms_with_data]

bp = ax.boxplot(box_data, labels=llms_with_data, patch_artist=True, widths=0.6,
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color="#666"), capprops=dict(color="#666"))

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Human baseline line
ax.axhline(y=human_mean, color="#ff5555", linestyle="--", alpha=0.8, linewidth=2, label=f"Human Corpus Mean ({human_mean:.2f})")
ax.axhspan(human_mean - human_std, human_mean + human_std, alpha=0.1, color="#ff5555", label="Human ±1 std")

ax.set_ylabel("RoBERTa Creativity Score", fontsize=12)
ax.set_title("LLM Creativity Scores vs Human Baseline", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/boxplot_roberta.png", dpi=200)
plt.close()

# 2. Radar Chart — Average Dimension Scores by LLM
dims = ["lexical_richness", "syntactic_complexity", "novelty", "imagery", "narrative_dynamics"]
dim_labels = ["Lexical\nRichness", "Syntactic\nComplexity", "Novelty", "Imagery", "Narrative\nDynamics"]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
angles += angles[:1]  # close the polygon

for llm in llms_with_data:
    means = df[df["llm"] == llm][dims].mean().tolist()
    means += means[:1]
    ax.plot(angles, means, "o-", linewidth=2, label=llm,
            color=LLM_COLORS.get(llm, "#888"), markersize=6)
    ax.fill(angles, means, alpha=0.1, color=LLM_COLORS.get(llm, "#888"))

# Human baseline
human_dim_means = [corpus_stats[d]["mean"] for d in dims]
human_dim_means += human_dim_means[:1]
ax.plot(angles, human_dim_means, "o--", linewidth=2, label="Human Corpus",
        color="#ff5555", markersize=6)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dim_labels, fontsize=10)
ax.set_ylim(0, 10)
ax.set_title("Creativity Dimensions — LLMs vs Human Baseline", fontsize=14,
             fontweight="bold", pad=30)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/radar_dimensions.png", dpi=200)
plt.close()

# 3. Bar Chart — Mean Scores with Error Bars
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(llms_with_data) + 1)
means = [df[df["llm"] == llm]["roberta_score"].mean() for llm in llms_with_data] + [human_mean]
stds = [df[df["llm"] == llm]["roberta_score"].std() for llm in llms_with_data] + [human_std]
bar_colors = [LLM_COLORS.get(llm, "#888") for llm in llms_with_data] + ["#ff5555"]
labels = llms_with_data + ["Human\n(Corpus)"]

bars = ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.8,
              edgecolor="#333", linewidth=1)

for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f"{mean:.2f}", ha="center", fontsize=11, fontweight="bold", color="#e0e0e0")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("RoBERTa Creativity Score", fontsize=12)
ax.set_title("Mean Creativity Score — LLMs vs Human Writers", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(means) + 1.5)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bar_comparison.png", dpi=200)
plt.close()

# 4. Heatmap — Per-Prompt Scores
fig, ax = plt.subplots(figsize=(14, 8))

pivot = df.pivot_table(index="prompt_id", columns="llm", values="roberta_score")
pivot = pivot.reindex(columns=[l for l in LLMS if l in pivot.columns])

im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=8)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=11)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f"P{idx}" for idx in pivot.index], fontsize=9)
ax.set_xlabel("LLM", fontsize=12)
ax.set_ylabel("Prompt", fontsize=12)
ax.set_title("Creativity Score Heatmap — Per Prompt × LLM", fontsize=14, fontweight="bold")

# Add score values
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, color="black" if val > 4 else "white")

plt.colorbar(im, ax=ax, label="RoBERTa Score", shrink=0.8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/heatmap_prompts.png", dpi=200)
plt.close()


# ── Summary Table ───────────────────────────────────────────
print("\n" + "=" * 70)
print("  LLM CREATIVITY COMPARISON — SUMMARY")
print("=" * 70)
print(f"  {'LLM':<15} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6} {'n':>4}")
print("  " + "-" * 50)

for llm in llms_with_data:
    subset = df[df["llm"] == llm]["roberta_score"]
    print(f"  {llm:<15} {subset.mean():>6.2f} {subset.std():>6.2f} "
          f"{subset.min():>6.2f} {subset.max():>6.2f} {len(subset):>4}")

print(f"  {'Human Corpus':<15} {human_mean:>6.2f} {human_std:>6.2f}    —      —    272k")
print("=" * 70)
print(f"\nPlots saved to {OUTPUT_DIR}/")
print("  - boxplot_roberta.png")
print("  - radar_dimensions.png")
print("  - bar_comparison.png")
print("  - heatmap_prompts.png")
print(f"\nScores saved to {OUTPUT_DIR}/llm_scores.csv")
print("\n✅ LLM evaluation complete!")
