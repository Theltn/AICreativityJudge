"""
Automated Creativity Scoring Rubric
====================================
Five dimensions scored 0–10, combined with weighted composite.

Dimensions:
  1. Lexical Richness   (20%)  — TTR + hapax legomena
  2. Syntactic Complexity (15%) — dependency parse depth
  3. Novelty             (25%)  — TF-IDF surprise vs corpus
  4. Imagery             (20%)  — sensory/concrete word density
  5. Narrative Dynamics  (20%)  — sentiment arc variance
"""

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter

# ── Bootstrap ──────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])  # NER not needed, saves time
except OSError:
    print("Warning: spacy model 'en_core_web_sm' not found.")
    nlp = None

analyzer = SentimentIntensityAnalyzer()

# Ensure NLTK data is available
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


# ── Sensory / Concrete Word Lexicon ───────────────────────────
# Curated list of high-imageability words across the five senses.
# Sources: MRC Psycholinguistic Database categories, Lancaster Sensorimotor Norms.
SENSORY_WORDS = set()

_SIGHT = [
    "bright", "dark", "glow", "glowing", "shimmer", "shimmering", "shadow",
    "shadows", "flash", "flashing", "sparkle", "sparkling", "gleam", "gleaming",
    "blinding", "dim", "hazy", "vivid", "pale", "crimson", "scarlet", "golden",
    "silver", "luminous", "radiant", "opaque", "translucent", "flickering",
    "blazing", "dazzling", "murky", "misty", "foggy", "glare", "glint",
    "twinkle", "beam", "ray", "illuminate", "silhouette", "blur", "blurry",
    "colorful", "faded", "iridescent", "lustrous", "shiny", "spotted",
    "striped", "transparent", "cloudy", "sunlit", "moonlit", "starlit",
    "bloody", "ashen", "jet", "ivory", "amber", "emerald", "sapphire",
]

_SOUND = [
    "whisper", "whispered", "roar", "roaring", "crash", "crashing", "bang",
    "echo", "echoing", "hum", "humming", "buzz", "buzzing", "scream",
    "screaming", "shriek", "murmur", "murmuring", "rumble", "rumbling",
    "click", "clicking", "snap", "snapping", "creak", "creaking", "thunder",
    "thundering", "chime", "ring", "ringing", "wail", "wailing", "sizzle",
    "sizzling", "crackle", "crackling", "rustle", "rustling", "clatter",
    "thud", "thump", "howl", "howling", "groan", "groaning", "hiss",
    "hissing", "squeak", "squeaking", "boom", "booming", "clang",
    "silence", "silent", "quiet", "loud", "deafening", "muffled",
]

_TOUCH = [
    "cold", "warm", "hot", "freezing", "burning", "smooth", "rough",
    "sharp", "soft", "hard", "wet", "dry", "sticky", "slippery", "damp",
    "moist", "tender", "prickly", "silky", "velvety", "coarse", "gritty",
    "slimy", "icy", "scorching", "lukewarm", "numb", "tingling", "stinging",
    "throbbing", "aching", "sore", "bruised", "calloused", "feathery",
    "fluffy", "rigid", "brittle", "flexible", "firm", "limp",
    "chilly", "crisp", "frosty", "sweltering",
]

_TASTE = [
    "sweet", "bitter", "sour", "salty", "savory", "spicy", "bland",
    "tangy", "tart", "rich", "metallic", "acidic", "sugary", "pungent",
    "delicious", "flavorful", "tasteless", "minty", "zesty",
]

_SMELL = [
    "fragrant", "stench", "aroma", "odor", "scent", "musty", "smoky",
    "rotten", "fresh", "pungent", "perfumed", "acrid", "earthy", "floral",
    "stale", "reek", "reeking", "whiff", "aromatic", "fetid",
]

for word_list in [_SIGHT, _SOUND, _TOUCH, _TASTE, _SMELL]:
    SENSORY_WORDS.update(w.lower() for w in word_list)


# ══════════════════════════════════════════════════════════════
#  DIMENSION 1: Lexical Richness                    (weight 20%)
# ══════════════════════════════════════════════════════════════
def lexical_richness(text):
    """
    Type-Token Ratio + Hapax Legomena ratio.
    Returns raw score (0–1 range roughly).
    """
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    if not tokens:
        return 0.0

    types = set(tokens)
    ttr = len(types) / len(tokens)

    counts = Counter(tokens)
    hapax = sum(1 for w, c in counts.items() if c == 1)
    hapax_ratio = hapax / len(tokens)

    # Combined: 50/50 blend
    return (ttr * 0.5) + (hapax_ratio * 0.5)


# ══════════════════════════════════════════════════════════════
#  DIMENSION 2: Syntactic Complexity                (weight 15%)
# ══════════════════════════════════════════════════════════════
def syntactic_complexity(text):
    """
    Average dependency parse depth + words-per-sentence factor.
    Returns raw score (roughly 1–10 range).
    """
    if nlp is None:
        return 0.0

    doc = nlp(text[:10000])

    depths = []
    for sent in doc.sents:
        def walk(node, depth):
            children = list(node.children)
            if not children:
                return depth
            return max(walk(child, depth + 1) for child in children)
        depths.append(walk(sent.root, 1))

    if not depths:
        return 0.0

    avg_depth = sum(depths) / len(depths)
    words_per_sent = len(doc) / max(1, len(list(doc.sents)))

    return avg_depth * 0.7 + (words_per_sent / 10.0) * 0.3


# ══════════════════════════════════════════════════════════════
#  DIMENSION 3: Novelty / TF-IDF Surprise           (weight 25%)
# ══════════════════════════════════════════════════════════════
def novelty_score(text, idf_dict=None, corpus_mean_idf=None):
    """
    Measures how 'surprising' the vocabulary is relative to the corpus.

    For each word in the text, look up its IDF (inverse document frequency).
    High-IDF words are rare across the corpus → more novel.
    The story's mean IDF is compared to the corpus-wide mean IDF.

    If idf_dict is None (corpus stats not yet computed), returns a
    placeholder based on hapax ratio as a rough proxy.
    """
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    if not tokens:
        return 0.0

    if idf_dict is not None and corpus_mean_idf is not None:
        # Real TF-IDF surprise: mean IDF of this story's tokens
        idf_values = [idf_dict.get(t, corpus_mean_idf) for t in tokens]
        story_mean_idf = np.mean(idf_values)
        # Ratio: how much more surprising than average
        surprise = story_mean_idf / max(corpus_mean_idf, 1e-9)
        return float(surprise)
    else:
        # Fallback: use unique-word ratio as rough proxy
        unique_ratio = len(set(tokens)) / len(tokens)
        return float(unique_ratio)


# ══════════════════════════════════════════════════════════════
#  DIMENSION 4: Imagery / Sensory Density           (weight 20%)
# ══════════════════════════════════════════════════════════════
def imagery_score(text):
    """
    Ratio of sensory/concrete words in the text.
    Uses a curated lexicon of ~200 high-imageability words
    spanning sight, sound, touch, taste, and smell.
    """
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    if not tokens:
        return 0.0

    sensory_count = sum(1 for t in tokens if t in SENSORY_WORDS)
    return sensory_count / len(tokens)   # ratio, typically 0.01–0.10


# ══════════════════════════════════════════════════════════════
#  DIMENSION 5: Narrative Dynamics                  (weight 20%)
# ══════════════════════════════════════════════════════════════
def narrative_dynamics(text):
    """
    Sentiment arc variance using VADER.
    Stories with bigger emotional swings → higher score.
    """
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 2:
        return 0.0

    sentiments = [analyzer.polarity_scores(s)["compound"] for s in sentences]
    variance = float(np.var(sentiments))
    return variance   # typically 0.0–0.5


# ══════════════════════════════════════════════════════════════
#  COMPOSITE SCORER
# ══════════════════════════════════════════════════════════════
WEIGHTS = {
    "lexical_richness":       0.20,
    "syntactic_complexity":   0.15,
    "novelty":                0.25,
    "imagery":                0.20,
    "narrative_dynamics":     0.20,
}

def score_story_raw(text, idf_dict=None, corpus_mean_idf=None):
    """
    Returns the 5 raw dimension scores for a single story.
    These are NOT yet normalized to 0–10.
    """
    return {
        "lexical_richness":     lexical_richness(text),
        "syntactic_complexity": syntactic_complexity(text),
        "novelty":              novelty_score(text, idf_dict, corpus_mean_idf),
        "imagery":              imagery_score(text),
        "narrative_dynamics":   narrative_dynamics(text),
    }


def normalize_scores(raw_scores_df):
    """
    Takes a DataFrame of raw scores and normalizes each column
    to 0–10 using percentile-based min-max scaling.

    Clips to [2nd, 98th] percentile to reduce outlier distortion.
    """
    normalized = raw_scores_df.copy()
    for col in WEIGHTS.keys():
        lo = raw_scores_df[col].quantile(0.02)
        hi = raw_scores_df[col].quantile(0.98)
        clipped = raw_scores_df[col].clip(lo, hi)
        if hi - lo > 0:
            normalized[col] = ((clipped - lo) / (hi - lo)) * 10.0
        else:
            normalized[col] = 5.0
    return normalized


def compute_composite(normalized_df):
    """
    Weighted sum of normalized dimension scores → composite 0–10.
    """
    composite = sum(
        normalized_df[dim] * weight
        for dim, weight in WEIGHTS.items()
    )
    return composite.clip(0, 10)


# ── Quick Test ─────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "The bright red sun dipped below the jagged horizon, painting the sky "
        "in streaks of crimson and gold. She felt a sudden, profound sadness "
        "wash over her like a cold wave. The silence was deafening. Then, from "
        "somewhere deep inside, a spark of hope ignited in her chest, warm and "
        "insistent. She whispered a promise to the fading light."
    )
    print("Testing scoring rubric on sample text:")
    scores = score_story_raw(sample)
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
