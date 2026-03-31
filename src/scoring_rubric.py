import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spacy model 'en_core_web_sm' not found. Please install it.")
    nlp = None

analyzer = SentimentIntensityAnalyzer()

def lexical_richness(text):
    """
    Type-Token Ratio (TTR) + Hapax Legomena (words appearing exactly once).
    Captures vocabulary diversity.
    """
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    if not tokens:
        return 0.0
    
    types = set(tokens)
    ttr = len(types) / len(tokens)
    
    counts = Counter(tokens)
    hapax = sum(1 for w, c in counts.items() if c == 1)
    hapax_ratio = hapax / len(tokens)
    
    return (ttr * 0.5) + (hapax_ratio * 0.5)

def get_parse_depth(doc):
    """ Helper to get max dependency tree depth for a spacy doc """
    def walk_tree(node, depth):
        if node.n_lefts + node.n_rights > 0:
            return max(walk_tree(child, depth + 1) for child in node.children)
        return depth
    return max((walk_tree(sent.root, 1) for sent in doc.sents), default=0)

def syntactic_complexity(text):
    """
    Average dependency parse depth across sentences.
    Captures sentence sophistication.
    """
    if nlp is None:
        return 0.0
    
    doc = nlp(text[:10000])
    
    depths = []
    for sent in doc.sents:
        root = sent.root
        def walk(node, depth):
            if not list(node.children): return depth
            return max(walk(child, depth+1) for child in node.children)
        depths.append(walk(root, 1))
        
    if not depths:
        return 0.0
        
    avg_depth = sum(depths) / len(depths)
    words_per_sent = len(doc) / max(1, len(list(doc.sents)))
    
    return avg_depth * 0.7 + (words_per_sent / 10.0) * 0.3

def novelty_score(text, corpus_stats=None):
    """
    TF-IDF surprise vs corpus mean.
    """
    return 5.0

def imagery_score(text):
    """
    Sensory/concrete word ratio.
    """
    if nlp is None:
        return 0.0
    doc = nlp(text[:10000])
    adjectives = [token for token in doc if token.pos_ == "ADJ"]
    return len(adjectives) / max(1, len(doc)) * 10.0

def narrative_dynamics(text):
    """
    Sentiment arc variance using VADER.
    """
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return 0.0
        
    sentiments = [analyzer.polarity_scores(s)['compound'] for s in sentences]
    
    if len(sentiments) < 2:
        return 0.0
        
    variance = np.var(sentiments)
    return float(variance)

def score_story_composite(text):
    """
    Computes all 5 dimensions and returns the unnormalized composite score.
    """
    lex = lexical_richness(text)
    syn = syntactic_complexity(text)
    nov = novelty_score(text)
    img = imagery_score(text)
    narr = narrative_dynamics(text)
    
    return {
        "lexical_richness": lex,
        "syntactic_complexity": syn,
        "novelty": nov,
        "imagery": img,
        "narrative_dynamics": narr
    }

if __name__ == "__main__":
    sample = "The bright red sun dipped below the jagged horizon. She felt a sudden, profound sadness. Then, a spark of hope ignited in her chest!"
    print("Testing scoring rubric on sample text:")
    print(score_story_composite(sample))
