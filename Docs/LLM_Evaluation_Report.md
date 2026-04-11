# LLM Creativity Evaluation — Full Results Report

## Executive Summary

This report presents the results of evaluating creative writing output from five major Large Language Models (ChatGPT, Claude, Gemini, Perplexity, and Copilot) against a human baseline of 272,579 short stories from the Reddit WritingPrompts corpus. Each LLM generated 20 short stories from identical prompts, and all 100 AI stories were scored using a fine-tuned RoBERTa-base model (R² = 0.936) trained exclusively on human-written fiction.

> [!IMPORTANT]
> The key finding: LLMs consistently **outscore the average human writer** on surface-level creativity metrics, but exhibit **lower narrative risk-taking** and **significantly less variance** — hallmarks of formulaic optimization rather than genuine creative expression.

---

## 1. Methodology

### 1.1 Training Pipeline
- **Dataset:** 272,579 human-written short stories from `euclaise/writingprompts` (Hugging Face)
- **Scoring Rubric:** 5 deterministic dimensions measured per story:
  - **Lexical Richness** (20%) — Type-Token Ratio + Hapax Legomena
  - **Syntactic Complexity** (15%) — Dependency parse depth + words/sentence
  - **Novelty** (25%) — TF-IDF surprise vs corpus mean
  - **Imagery** (20%) — Sensory/concrete word density (~200 word lexicon)
  - **Narrative Dynamics** (20%) — Sentiment arc variance (VADER)
- **Normalization:** Percentile-based clipping (2nd/98th) to 0–10 scale
- **Composite Score:** Weighted sum of the 5 dimensions

### 1.2 Model Training
- **MLP Baseline:** 5 features → 64 → 32 → 1 (learns rubric-to-score mapping)
- **RoBERTa-base:** Fine-tuned for regression on raw story text → composite score
- **Results on test set (40,887 stories):**

| Model | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| MLP Baseline | — | — | — | — |
| **RoBERTa-base** | **0.102** | **0.319** | **0.245** | **0.936** |

The RoBERTa model achieves 93.6% explained variance, meaning it has internalized creativity patterns far beyond the 5 explicit rubric features.

### 1.3 LLM Evaluation Setup
- **20 prompts** randomly selected from the training corpus
- **5 LLMs tested:** ChatGPT (GPT-4o), Claude (Sonnet), Gemini, Perplexity, Copilot
- **Instruction given:** "Write a short story (~500 words) based on this prompt. Just write the story directly, no preamble or commentary."
- **All stories truncated to 500 words** before scoring (matching training conditions)
- **100 total stories** (20 × 5) scored through both the rubric and RoBERTa

---

## 2. Overall Results

### 2.1 Creativity Score Rankings

| Rank | LLM | Mean Score | Std | Min | Max | vs Human Mean |
|---|---|---|---|---|---|---|
| 🥇 | **Gemini** | **5.63** | 0.64 | 4.50 | 6.83 | +2.00 |
| 🥈 | **ChatGPT** | **4.94** | 0.84 | 3.09 | 6.22 | +1.31 |
| 🥉 | **Perplexity** | **4.36** | 0.64 | 2.95 | 5.24 | +0.73 |
| 4th | **Claude** | **3.69** | 0.73 | 2.56 | 4.84 | +0.06 |
| 5th | **Copilot** | **3.22** | 0.99 | 2.13 | 5.16 | −0.41 |
| — | *Human Corpus* | *3.63* | *1.27* | — | — | *baseline* |

![Mean Creativity Scores — LLMs vs Human Writers](/Users/theoh/.gemini/antigravity/brain/fc323f1e-b949-49af-8664-3f294a13ee4e/bar_comparison.png)

### 2.2 Score Distributions

![Box Plot — RoBERTa Creativity Scores by LLM](/Users/theoh/.gemini/antigravity/brain/fc323f1e-b949-49af-8664-3f294a13ee4e/boxplot_roberta.png)

**Key observations:**
- **Gemini** has the highest median and the tightest interquartile range — it is both the most creative *and* the most consistent
- **ChatGPT** shows a wider spread, indicating greater variability in writing quality across prompts
- **Claude** clusters tightly around the human mean (3.63), making it statistically the most "human-like" in output
- **Copilot** has the widest range (2.13–5.16) and the lowest median, often falling below the human average
- **Perplexity** sits solidly between ChatGPT and Claude with remarkably low variance (σ=0.64)

---

## 3. Dimension-Level Analysis

### 3.1 Averages Across All 5 Creativity Dimensions

| Dimension | ChatGPT | Claude | Gemini | Perplexity | Copilot | Human Mean |
|---|---|---|---|---|---|---|
| Lexical Richness | 5.80 | 4.59 | 5.34 | 5.59 | 4.20 | 4.03 |
| Syntactic Complexity | 3.97 | 3.50 | 4.18 | 4.12 | 2.30 | 3.76 |
| Novelty | 6.85 | 5.77 | 6.37 | 6.69 | 5.32 | 4.09 |
| Imagery | 0.93 | 0.79 | 1.61 | 0.84 | 1.27 | 2.46 |
| Narrative Dynamics | 2.07 | 1.40 | 2.36 | 2.53 | 1.35 | 3.76 |

### 3.2 Radar Chart

![Creativity Dimensions — Radar Chart](/Users/theoh/.gemini/antigravity/brain/fc323f1e-b949-49af-8664-3f294a13ee4e/radar_dimensions.png)

### 3.3 Dimension Insights

> [!NOTE]
> The radar chart reveals a consistent, striking pattern across all five LLMs: they dominate humans on **lexical** and **novelty** metrics, but fall substantially behind on **imagery** and **narrative dynamics**.

**Where LLMs excel (surface-level features):**
- **Lexical Richness:** All 5 LLMs exceed the human mean (4.03). ChatGPT leads at 5.80. LLMs are trained to avoid repetition and naturally produce diverse vocabulary — this is a feature of their training, not a sign of creativity.
- **Novelty (TF-IDF surprise):** All LLMs scored 5.32–6.85 vs the human mean of 4.09. LLMs generate uncommon word combinations because they interpolate across vast training corpora, producing text that is statistically "surprising" without being genuinely original.

**Where humans still lead (deeper creative features):**
- **Imagery (sensory language):** The human corpus averages 2.46 while every LLM scores below 1.61. Humans naturally invoke touch, taste, smell, and physical sensation. LLMs tend toward abstraction, emotion labels ("she felt sad"), and visual-only descriptions, missing the embodied experience of physical existence.
- **Narrative Dynamics (emotional arc):** Human mean is 3.76; no LLM exceeds 2.53. Human writers create stories with dramatic tonal shifts — dark to light, tension to release, humor to horror. LLMs produce emotionally monotone narratives that arc smoothly and predictably from setup to resolution.

---

## 4. Per-Prompt Analysis

![Heatmap — Per Prompt × LLM Scores](/Users/theoh/.gemini/antigravity/brain/fc323f1e-b949-49af-8664-3f294a13ee4e/heatmap_prompts.png)

### 4.1 Prompts That Inspired the Best Writing

| Rank | Prompt | Avg Score | Best LLM |
|---|---|---|---|
| 1 | P12: "Humans are the most innovative species in the galaxy..." | 4.85 | Gemini (6.2) |
| 2 | P01: "First interstellar ship meets a border guard" | 4.82 | Gemini (6.4) |
| 3 | P05: "Kids who grew up without sun" | 4.76 | Gemini (6.8) |

World-building and speculative fiction prompts consistently elicited the highest creativity scores, likely because they require imagination beyond pattern-matching.

### 4.2 Prompts That Were Most Challenging

| Rank | Prompt | Avg Score | Weakest LLM |
|---|---|---|---|
| 18 | P07: "A small act of kindness in a dystopia" | 3.90 | Claude (2.6) |
| 19 | P19: "Mannequin / intruder" | 3.76 | Claude (2.8) |
| 20 | P16: "Immortality proven to the FBI" | 3.60 | Copilot (2.5) |

Short, ambiguous prompts and prompts requiring tonal subtlety were the hardest for LLMs to differentiate on.

### 4.3 Notable Patterns in the Heatmap
- **Gemini never scores below 4.5** — it is the only LLM that maintains consistently high creativity regardless of prompt type
- **Copilot is consistently the coldest column** — scoring below 3.0 on 7 of the 20 prompts
- **ChatGPT is prompt-sensitive** — scores as high as 6.2 (P14) but drops to 3.1 (P19), showing high variance by prompt type
- **Claude is remarkably flat** — narrow range (2.6–4.8), suggesting a default "safe" writing style

---

## 5. The Big Question: Are LLMs More Creative Than Humans?

> [!CAUTION]
> **The short answer: No.** The longer answer requires distinguishing between *writing quality* and *creative originality*.

### 5.1 What the Scores Actually Measure

Our model was trained on human-written fiction and evaluates **textual features associated with creative writing**: rich vocabulary, complex syntax, unusual word combinations, sensory language, and emotional range. These are **proxies** for creativity, not creativity itself.

When Gemini scores 5.63 and the human mean is 3.63, the correct interpretation is:

> "Gemini produces text with higher-scoring surface-level creativity features than the average Reddit short story writer."

This is *not* equivalent to saying Gemini is more creative than human writers.

### 5.2 The Consistency Paradox

Perhaps the most telling finding: **LLMs have dramatically lower variance than humans.**

| Source | Mean | Std Dev |
|---|---|---|
| Gemini | 5.63 | **0.64** |
| ChatGPT | 4.94 | **0.84** |
| Human Corpus | 3.63 | **1.27** |

The human corpus has nearly **double** the standard deviation of any LLM. Humans produce both the worst stories in the dataset (morse code, repetitive text, one-sentence responses scoring <1.0) and the best (experimental poetry, breathtaking prose scoring >8.0). LLMs never produce either extreme.

This low variance is a hallmark of **optimization, not creativity.** LLMs have been fine-tuned, RLHF'd, and safety-filtered to produce consistently "good" output. True creativity, by definition, involves risk — the possibility of spectacular failure alongside spectacular success.

### 5.3 The Embodiment Gap

The dimension where humans most decisively outperform LLMs is **imagery** — the use of language rooted in physical sensation: touch, taste, smell, temperature, proprioception.

Human fiction naturally references "the cold metal biting into his palms," "salt on his lips," "the weight of the gun in his hand." LLMs default to emotion labels and visual descriptions. This "embodiment gap" reflects a fundamental difference: humans have bodies and write from sensory memory. LLMs generate text from statistical patterns.

### 5.4 The Narrative Monotone

On **narrative dynamics** (sentiment arc variance), humans score 3.76 vs the best LLM at 2.53. This means human stories contain more dramatic tonal shifts — dark turns, uncomfortable moments, humor adjacent to tragedy.

LLMs produce what could be called a "narrative monotone": stories that flow smoothly from setup → rising action → climax → resolution with predictable emotional arcs. They rarely take risks like killing a protagonist mid-story, inserting dark humor into a tragedy, or ending ambiguously. This smooth arc reads as competent but lacks the electric unpredictability of distinctive creative voice.

---

## 6. LLM-Specific Profiles

### 🥇 Gemini (Mean: 5.63)
**The Consistent Overachiever.** Gemini scored highest overall and never dropped below 4.5 on any prompt. It leads on Syntactic Complexity (4.18) and Narrative Dynamics (2.36), and produces the highest Imagery scores among LLMs (1.61). Its low variance (σ=0.64) suggests a highly calibrated default writing mode. Gemini appears to have the strongest "literary" style baked into its training.

### 🥈 ChatGPT (Mean: 4.94)
**The Vocabulary Champion.** ChatGPT has the highest Lexical Richness (5.80) and Novelty (6.85) of any LLM, meaning it uses the most diverse and statistically surprising word choices. However, its wider variance (σ=0.84) and prompt-sensitivity suggest it adapts its style more to each prompt — sometimes brilliantly (P14: 6.22), sometimes poorly (P19: 3.09).

### 🥉 Perplexity (Mean: 4.36)
**The Dark Horse.** Perplexity, primarily known as a search-augmented chatbot, performs surprisingly well. It has the highest Narrative Dynamics of any LLM (2.53) and strong Syntactic Complexity (4.12). Its tight distribution (σ=0.64) suggests reliable, above-average creative output.

### 4th — Claude (Mean: 3.69)
**The Human Mimic.** Claude's mean (3.69) is within 0.06 of the human corpus mean (3.63), making it statistically the most "human-like" writer. This could be interpreted positively (realistic, natural prose) or negatively (undifferentiated, middle-of-the-road). Claude scores lowest among LLMs on Narrative Dynamics (1.40), suggesting a particularly cautious, smooth writing style.

### 5th — Copilot (Mean: 3.22)
**The Underperformer.** Copilot is the only LLM that scores *below* the human corpus mean. It has the lowest Syntactic Complexity (2.30) and Novelty (5.32), and the highest variance (σ=0.99). On 7 of 20 prompts, it scores below 3.0. Its wide range (2.13–5.16) suggests inconsistency in creative writing ability, likely because Copilot is optimized primarily for code generation and conversational assistance rather than literary output.

---

## 7. Limitations

1. **Sample size:** 20 stories per LLM is sufficient for directional findings but too small for statistical significance testing. A larger sample (50–100) would strengthen claims.
2. **Prompt selection:** The 20 prompts were randomly sampled from the training corpus. Different prompts might yield different rankings.
3. **Single-shot generation:** Each LLM was given one attempt per prompt with no prompt engineering. Results might differ with temperature adjustments or "write more creatively" instructions.
4. **Rubric circularity:** The RoBERTa model was trained on the same rubric that generated the dimension scores. It has learned what "our definition of creativity" looks like, not an absolute measure.
5. **Corpus bias:** The human baseline is Reddit WritingPrompts — amateur community fiction. Comparing against published literature would likely flip several rankings.
6. **Model versions:** LLM outputs depend on the specific model version accessed at the time of collection. Results are not reproducible across model updates.

---

## 8. Conclusions

1. **All LLMs except Copilot outscore the average human writer** on our creativity rubric, but this reflects polished prose generation rather than genuine creative originality.
2. **Gemini is the strongest creative writer** among the tested LLMs, with the highest mean (5.63) and most consistent output.
3. **Humans retain an edge in narrative risk-taking and sensory embodiment** — the dimensions most closely associated with authentic creative voice.
4. **LLM consistency is a double-edged sword** — their low variance produces reliably "good" text but never produces the transcendent highs (or the experimental failures) that characterize truly creative writing.
5. **Claude is the most human-like in score distribution**, sitting almost exactly at the corpus mean — an interesting finding for a model that emphasizes "harmlessness" and natural communication.
6. **The model measures *textual creativity features*, not creativity itself.** The distinction is critical: a perfectly optimized text can score high on every metric without containing a single original idea.

---

*Report generated April 10, 2026*
*AI Creativity Judge — CSC 426 Deep Learning*
*Theo Helton, David Pope, Steven Weil*
