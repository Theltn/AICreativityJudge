"""
Creativity Evaluation API
==========================
FastAPI server that loads the trained RoBERTa model and rubric scoring
functions to evaluate text creativity in real-time.

Run: python src/api.py
"""

import os
import sys
import json
import time
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.scoring_rubric import (
    lexical_richness,
    syntactic_complexity,
    novelty_score,
    imagery_score,
    narrative_dynamics,
)

# ── App Setup ──────────────────────────────────────────────────
app = FastAPI(title="AI Creativity Judge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals (loaded on startup) ────────────────────────────────
model = None
tokenizer = None
device = None
corpus_stats = None

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models", "roberta_creativity_model")
STATS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "models", "corpus_stats.json")


@app.on_event("startup")
def load_model():
    global model, tokenizer, device, corpus_stats

    print("Loading RoBERTa model...")
    t0 = time.time()

    # Use MPS on Mac, CUDA on PC, fallback to CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("cpu")  # MPS can be flaky for inference, CPU is fine
    else:
        device = torch.device("cpu")

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
    model = model.to(device)
    model.eval()

    # Load corpus stats for context
    with open(STATS_PATH) as f:
        corpus_stats = json.load(f)

    print(f"Model loaded on {device} in {time.time()-t0:.1f}s")


# ── Request/Response Models ────────────────────────────────────
class EvaluateRequest(BaseModel):
    text: str


class DimensionScore(BaseModel):
    score: float
    percentile_context: str  # e.g. "above average", "top 10%"


class EvaluateResponse(BaseModel):
    composite_score: float
    roberta_score: float
    dimensions: dict
    word_count: int
    verdict: str
    processing_time_ms: float


# ── Helpers ────────────────────────────────────────────────────
def truncate(text, max_words=500):
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def get_context(value, dim_stats):
    """Return human-readable context + direction for a score vs the corpus."""
    mean = dim_stats["mean"]
    p10 = dim_stats["p10"]
    p90 = dim_stats["p90"]
    diff = value - mean

    if value >= p90:
        return {"label": "Top 10%", "direction": "up", "vs_mean": round(diff, 1)}
    elif value >= mean * 1.2:
        return {"label": "Above Avg", "direction": "up", "vs_mean": round(diff, 1)}
    elif value >= mean * 0.8:
        return {"label": "Average", "direction": "neutral", "vs_mean": round(diff, 1)}
    elif value >= p10:
        return {"label": "Below Avg", "direction": "down", "vs_mean": round(diff, 1)}
    else:
        return {"label": "Bottom 10%", "direction": "down", "vs_mean": round(diff, 1)}


def roberta_predict(text):
    """Run RoBERTa inference on a single text."""
    inputs = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model(**inputs)
        score = output.logits.squeeze().item()

    # Clip to 0-10 range
    return max(0.0, min(10.0, score))


def get_verdict(score):
    """
    Verdict calibrated to the actual corpus distribution:
      Mean: 3.63  |  Std: 1.27  |  P10: 2.09  |  P90: 5.32

    A score of 5.0 is better than ~85% of 272k human stories.
    A score of 6.0+ is exceptionally rare.
    """
    if score >= 5.5:
        return "Exceptional Creativity (Top 5%) — This text demonstrates outstanding creative writing. Vocabulary, structure, and imagery far exceed the typical story in our 272k corpus."
    elif score >= 4.5:
        return "High Creativity (Top 15%) — Strong creative indicators. Rich vocabulary and engaged narrative structure place this well above the average human-written fiction."
    elif score >= 3.5:
        return "Average Creativity — Comparable to the median human-written short story in our corpus. Solid writing with room for more distinctive voice."
    elif score >= 2.5:
        return "Below Average Creativity — Limited vocabulary diversity and simpler sentence structure. The writing follows predictable patterns."
    else:
        return "Low Creativity (Bottom 10%) — Repetitive structure, minimal vocabulary variation, and flat narrative arc. Formulaic or template-driven writing."



# ── Main Endpoint ──────────────────────────────────────────────
@app.post("/api/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest):
    t0 = time.time()

    text = truncate(request.text.strip(), 500)
    word_count = len(text.split())

    # 1. RoBERTa composite prediction
    rob_score = roberta_predict(text)

    # 2. Rubric dimension scores (raw)
    lex_raw = lexical_richness(text)
    syn_raw = syntactic_complexity(text)
    nov_raw = novelty_score(text)  # fallback mode (no IDF dict)
    img_raw = imagery_score(text)
    nar_raw = narrative_dynamics(text)

    # 3. Normalize each dimension to 0-10 using corpus percentiles
    # We'll use a simple mapping based on known corpus ranges
    def normalize(raw, dim_name):
        stats = corpus_stats[dim_name]
        # Map the raw score into 0-10 relative to corpus
        # Using the mean and std to create a z-score based normalization
        z = (raw - 0) / max(stats["std"] / 10 * 2, 1e-9)  # rough scale
        return max(0.0, min(10.0, z))

    # More practical: just scale based on observed raw ranges
    # Lexical: raw ~0.3-0.9 → 0-10
    lex_norm = max(0, min(10, (lex_raw - 0.2) / 0.6 * 10))
    # Syntactic: raw ~1-8 → 0-10
    syn_norm = max(0, min(10, (syn_raw - 1.0) / 6.0 * 10))
    # Novelty (fallback): raw ~0.3-0.9 → 0-10
    nov_norm = max(0, min(10, (nov_raw - 0.2) / 0.6 * 10))
    # Imagery: raw ~0.0-0.15 → 0-10
    img_norm = max(0, min(10, img_raw / 0.12 * 10))
    # Narrative: raw ~0.0-0.5 → 0-10
    nar_norm = max(0, min(10, nar_raw / 0.4 * 10))

    # Weighted rubric composite
    rubric_composite = (
        0.20 * lex_norm +
        0.15 * syn_norm +
        0.25 * nov_norm +
        0.20 * img_norm +
        0.20 * nar_norm
    )
    rubric_composite = max(0, min(10, rubric_composite))

    # Use RoBERTa as the primary score, rubric for the breakdown
    dimensions = {
        "lexical_richness": {
            "score": round(lex_norm, 2),
            "context": get_context(lex_norm, corpus_stats["lexical_richness"]),
        },
        "syntactic_complexity": {
            "score": round(syn_norm, 2),
            "context": get_context(syn_norm, corpus_stats["syntactic_complexity"]),
        },
        "novelty": {
            "score": round(nov_norm, 2),
            "context": get_context(nov_norm, corpus_stats["novelty"]),
        },
        "imagery": {
            "score": round(img_norm, 2),
            "context": get_context(img_norm, corpus_stats["imagery"]),
        },
        "narrative_dynamics": {
            "score": round(nar_norm, 2),
            "context": get_context(nar_norm, corpus_stats["narrative_dynamics"]),
        },
    }

    elapsed_ms = (time.time() - t0) * 1000

    return EvaluateResponse(
        composite_score=round(rubric_composite, 2),
        roberta_score=round(rob_score, 2),
        dimensions=dimensions,
        word_count=word_count,
        verdict=get_verdict(rob_score),
        processing_time_ms=round(elapsed_ms, 0),
    )


# ── Health Check ───────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
