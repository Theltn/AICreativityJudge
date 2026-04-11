# 🧠 AI Creativity Judge

**Automated evaluation of creative writing using deep learning.**

A fine-tuned RoBERTa model trained on 272,579 human-written short stories to predict creativity scores on a 0–10 scale. Includes a real-time web interface for scoring individual stories and a comparative evaluation of 5 major LLMs (ChatGPT, Claude, Gemini, Perplexity, Copilot).

**CSC 426 — Deep Learning | Spring 2026**
Theo Helton, David Pope, Steven Weil

---

## Table of Contents

- [Overview](#overview)
- [Results at a Glance](#results-at-a-glance)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation — macOS](#installation--macos)
- [Installation — Windows](#installation--windows)
- [Running the Full Pipeline](#running-the-full-pipeline)
  - [Step 1: Data Collection](#step-1-data-collection)
  - [Step 2: Preprocessing & Scoring](#step-2-preprocessing--scoring)
  - [Step 3: Train/Val/Test Split](#step-3-trainvaltest-split)
  - [Step 4: Model Training](#step-4-model-training)
  - [Step 5: Run the Web App](#step-5-run-the-web-app)
  - [Step 6: LLM Story Evaluation](#step-6-llm-story-evaluation)
- [Notebooks](#notebooks)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project measures creative writing quality using a 5-dimension rubric:

| Dimension | Weight | What it Measures |
|---|---|---|
| **Lexical Richness** | 20% | Type-Token Ratio + Hapax Legomena (unique words) |
| **Syntactic Complexity** | 15% | Dependency parse depth + sentence length variance |
| **Novelty** | 25% | TF-IDF surprise vs corpus average |
| **Imagery** | 20% | Sensory/concrete word density (~200 word lexicon) |
| **Narrative Dynamics** | 20% | Sentiment arc variance (VADER) |

A RoBERTa-base transformer is fine-tuned to predict the weighted composite of these 5 scores directly from raw text, achieving **R² = 0.936** on the test set.

---

## Results at a Glance

**Model Performance (Test Set — 40,887 stories):**

| Metric | Value |
|---|---|
| R² | 0.936 |
| RMSE | 0.319 |
| MAE | 0.245 |

**LLM Creativity Rankings (20 stories each, same prompts):**

| Rank | LLM | Mean Score | vs Human (3.63) |
|---|---|---|---|
| 🥇 | Gemini | 5.63 | +2.00 |
| 🥈 | ChatGPT | 4.94 | +1.31 |
| 🥉 | Perplexity | 4.36 | +0.73 |
| 4th | Claude | 3.69 | +0.06 |
| 5th | Copilot | 3.22 | −0.41 |

> See [Docs/LLM_Evaluation_Report.md](Docs/LLM_Evaluation_Report.md) for the full analysis.

---

## Project Structure

```
AICreativityJudge/
├── README.md                      ← You are here
├── requirements.txt               ← Python dependencies
│
├── src/
│   ├── api.py                     ← FastAPI backend (loads model, serves scores)
│   ├── scoring_rubric.py          ← 5-dimension creativity rubric
│   └── data_preprocessing.py      ← Text cleaning / filtering
│
├── scripts/
│   ├── data_collection.py         ← Downloads dataset from Hugging Face
│   ├── score_dataset.py           ← Scores all 272k stories with rubric
│   ├── sanity_check_and_split.py  ← Validates scores, creates train/val/test
│   ├── train_local.py             ← Local GPU training (MLP + RoBERTa)
│   ├── score_llm_stories.py       ← Evaluates LLM stories + generates plots
│   ├── generate_eda_nb.py         ← Generates EDA notebook
│   ├── generate_training_nbs.py   ← Generates training notebooks
│   └── generate_pdf.py            ← Generates PDF report
│
├── notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_MLP_Baseline.ipynb
│   └── 03_RoBERTa_Finetuning.ipynb
│
├── frontend/                      ← React + Vite web interface
│   ├── src/App.jsx                ← Main UI component
│   ├── src/index.css              ← Cyberpunk/Neo-dark theme
│   └── package.json               ← Node dependencies
│
├── data/                          ← NOT in Git (too large, generated locally)
│   ├── raw/                       ← Downloaded parquets
│   ├── processed/                 ← Scored + split data
│   ├── models/                    ← Trained model weights
│   │   ├── roberta_creativity_model/  ← Main model (config, weights, tokenizer)
│   │   ├── corpus_stats.json          ← Normalization statistics
│   │   └── *.pt, *.png               ← Checkpoints, plots
│   └── llm_stories/               ← 100 LLM stories + results
│       ├── ChatGPT/, Claude/, Gemini/, Perplexity/, Copilot/
│       └── results/               ← Scores CSV + comparison plots
│
└── Docs/
    ├── LLM_Evaluation_Report.md   ← Full LLM comparison analysis
    ├── Project_Plan.md            ← Original project roadmap
    ├── Colab_Training_Instructions.md
    └── PC_Training_Setup.md
```

> **Note:** The `data/` directory is gitignored because it contains large files (4+ GB). You must generate it yourself by following the pipeline steps below, or obtain a copy from the team.

---

## Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| **Python** | 3.10–3.12 | Backend + training scripts |
| **Node.js** | 18+ | Frontend dev server |
| **Git** | Any | Clone the repo |
| **NVIDIA GPU** (optional) | CUDA 12.x | Speeds up training from hours → minutes |

---

## Installation — macOS

### 1. Install Homebrew (if not installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Python and Node.js
```bash
brew install python@3.12 node
```

### 3. Clone the Repository
```bash
git clone https://github.com/Theltn/AICreativityJudge.git
cd AICreativityJudge
```

### 4. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 5. Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Install PyTorch (CPU — Mac does not use CUDA)
```bash
pip install torch torchvision torchaudio
```

> **Apple Silicon (M1/M2/M3/M4):** PyTorch supports MPS acceleration automatically. The API runs on CPU for reliability, but training can use MPS.

### 7. Download NLP Models
```bash
# spaCy English model (required for syntactic complexity scoring)
python -m spacy download en_core_web_sm

# NLTK data (required for VADER sentiment + tokenization)
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon'); nltk.download('punkt_tab')"
```

### 8. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

✅ **macOS installation complete.**

---

## Installation — Windows

### 1. Install Python 3.12
Download from [python.org/downloads](https://www.python.org/downloads/). During installation:
- ✅ Check **"Add Python to PATH"**
- ✅ Check **"Install pip"**

### 2. Install Node.js 18+
Download from [nodejs.org](https://nodejs.org/) (LTS version).

### 3. Install Git
Download from [git-scm.com](https://git-scm.com/downloads). Use default settings.

### 4. Clone the Repository
Open **Command Prompt** or **PowerShell**:
```cmd
git clone https://github.com/Theltn/AICreativityJudge.git
cd AICreativityJudge
```

### 5. Create Virtual Environment
```cmd
python -m venv venv
venv\Scripts\activate
```

### 6. Install Python Dependencies
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

### 7. Install PyTorch

**With NVIDIA GPU (CUDA 12.6):**
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Without GPU (CPU only):**
```cmd
pip install torch torchvision torchaudio
```

> **Verify CUDA:** Run `python -c "import torch; print(torch.cuda.is_available())"` — should print `True` if GPU is working.

### 8. Download NLP Models
```cmd
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon'); nltk.download('punkt_tab')"
```

### 9. Install Frontend Dependencies
```cmd
cd frontend
npm install
cd ..
```

✅ **Windows installation complete.**

---

## Running the Full Pipeline

These steps reproduce the entire project from scratch. Each step creates output in `data/` that the next step depends on.

### Step 1: Data Collection

Downloads the WritingPrompts dataset (~272k stories) from Hugging Face.

```bash
python scripts/data_collection.py
```

**Output:** `data/raw/writing_prompts_validation.parquet`, `data/raw/writing_prompts_test.parquet`

Then download the full training set:
```bash
python -c "
from datasets import load_dataset
import pandas as pd, os
os.makedirs('data/raw', exist_ok=True)
ds = load_dataset('euclaise/writingprompts', split='train')
ds.to_pandas().to_parquet('data/raw/writing_prompts_full.parquet')
print(f'Downloaded {len(ds)} stories')
"
```

**Output:** `data/raw/writing_prompts_full.parquet` (~272k stories)

### Step 2: Preprocessing & Scoring

Cleans the text (removes URLs, markdown, Reddit tags) and scores every story on the 5-dimension rubric.

```bash
# Clean the raw dataset
python src/data_preprocessing.py --input data/raw/writing_prompts_full.parquet --output data/processed/writing_prompts_full_cleaned.parquet

# Score all stories (takes ~2-4 hours on first run)
python scripts/score_dataset.py
```

**Output:** `data/processed/writing_prompts_scored.parquet`

> **Note:** `score_dataset.py` supports checkpointing. If it's interrupted, just re-run it and it will resume.

### Step 3: Train/Val/Test Split

Validates score distributions and creates 70/15/15 train/val/test splits.

```bash
python scripts/sanity_check_and_split.py
```

**Output:** `data/processed/train.parquet`, `data/processed/val.parquet`, `data/processed/test.parquet`

### Step 4: Model Training

Trains both an MLP baseline and the RoBERTa model.

**Option A: Train Locally (requires NVIDIA GPU)**

```bash
python scripts/train_local.py
```

This trains:
1. **MLP Baseline** (~2 minutes) — 5 rubric features → score
2. **RoBERTa-base** (~20 min on RTX 5070, ~2 hrs on RTX 3060) — raw text → score

**Output:**
- `data/models/mlp_best.pt` — MLP weights
- `data/models/roberta_best.pt` — RoBERTa checkpoint
- `data/models/roberta_creativity_model/` — Saved model (config, weights, tokenizer)
- `data/models/corpus_stats.json` — Normalization statistics
- `data/models/*_scatter.png`, `*_training_curves.png` — Training plots

> **Training Time Estimates:**
> | Hardware | RoBERTa Training Time |
> |---|---|
> | RTX 5070 / 4080 | ~15-20 minutes |
> | RTX 3060 / 3070 | ~1-2 hours |
> | RTX 2060 | ~3-4 hours |
> | M4 MacBook (CPU) | ~8-12 hours |
> | CPU only (Intel/AMD) | ~24+ hours |

**Option B: Train on Google Colab (free GPU)**

See [Docs/Colab_Training_Instructions.md](Docs/Colab_Training_Instructions.md) for step-by-step instructions. Upload the notebooks from `notebooks/` to Colab and run them there.

### Step 5: Run the Web App

The web app has two parts: a **Python API** (backend) and a **React app** (frontend).

**Terminal 1 — Start the API:**
```bash
# macOS
source venv/bin/activate && python src/api.py

# Windows
venv\Scripts\activate && python src/api.py
```

You should see:
```
Loading RoBERTa model...
Model loaded on cpu in 0.2s
Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 — Start the Frontend:**
```bash
cd frontend
npm run dev
```

You should see:
```
VITE ready in 300ms
➜ Local: http://localhost:5173/
```

**Open http://localhost:5173 in your browser.** Paste any text and click EVALUATE to get real-time creativity scoring.

### Step 6: LLM Story Evaluation

After training, you can evaluate and compare LLM-generated stories.

**Collecting stories (manual):**
1. See `data/llm_stories/prompts.txt` for the 20 prompts
2. Paste each prompt into 5 LLMs (ChatGPT, Claude, Gemini, Perplexity, Copilot)
3. Save each response as `data/llm_stories/[LLM_NAME]/[NUMBER].txt`
   - Example: `data/llm_stories/ChatGPT/01.txt`
4. See `data/llm_stories/README.md` for detailed instructions

**Scoring and generating plots:**
```bash
python scripts/score_llm_stories.py
```

**Output:**
- `data/llm_stories/results/llm_scores.csv` — All scores
- `data/llm_stories/results/bar_comparison.png` — Mean scores bar chart
- `data/llm_stories/results/boxplot_roberta.png` — Score distributions
- `data/llm_stories/results/radar_dimensions.png` — 5-dimension radar chart
- `data/llm_stories/results/heatmap_prompts.png` — Per-prompt × per-LLM heatmap

---

## Notebooks

The Jupyter notebooks in `notebooks/` document the analysis and can be run independently:

| Notebook | Purpose |
|---|---|
| `01_Exploratory_Data_Analysis.ipynb` | Dataset statistics, score distributions, correlation analysis |
| `02_MLP_Baseline.ipynb` | MLP baseline model training and evaluation |
| `03_RoBERTa_Finetuning.ipynb` | RoBERTa fine-tuning, training curves, test results |

To run notebooks locally:
```bash
source venv/bin/activate   # or venv\Scripts\activate on Windows
jupyter notebook
```

Or upload to [Google Colab](https://colab.research.google.com/) for free GPU access.

---

## Documentation

| Document | Description |
|---|---|
| [LLM_Evaluation_Report.md](Docs/LLM_Evaluation_Report.md) | Full analysis of 5 LLMs vs human baseline |
| [Project_Plan.md](Docs/Project_Plan.md) | Original project roadmap and milestones |
| [Colab_Training_Instructions.md](Docs/Colab_Training_Instructions.md) | How to train on Google Colab |
| [PC_Training_Setup.md](Docs/PC_Training_Setup.md) | Windows/NVIDIA setup guide |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`
You need to install PyTorch separately. See the installation section for your platform above.

### `ModuleNotFoundError: No module named 'fastapi'`
Run `pip install fastapi uvicorn[standard] pydantic`

### `OSError: [E050] Can't find model 'en_core_web_sm'`
Run `python -m spacy download en_core_web_sm`

### `Resource punkt not found` / `Resource vader_lexicon not found`
Run `python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon'); nltk.download('punkt_tab')"`

### CUDA / GPU Not Detected on Windows
1. Verify NVIDIA drivers: `nvidia-smi` (should show your GPU)
2. Verify CUDA toolkit: `nvcc --version`
3. Uninstall and reinstall PyTorch with CUDA:
```cmd
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Frontend: `npm run dev` fails
Make sure Node.js 18+ is installed (`node --version`). Then:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### API: Port 8000 already in use
Kill the existing process:
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Training is very slow
- **No GPU:** Training on CPU takes 12-24+ hours. Use Google Colab (free T4 GPU) or a machine with an NVIDIA GPU.
- **Low GPU memory:** Reduce batch size in `scripts/train_local.py` (look for `BATCH_SIZE`).
- **MPS on Mac:** MPS acceleration is available but can be unstable. The scripts default to CPU for reliability.

### `data/` folder is empty / missing
The `data/` directory is gitignored because it's too large (~4+ GB). You must either:
1. **Generate it yourself** by running Steps 1–4 of the pipeline above
2. **Copy it** from a team member who has already run the pipeline

---

## License

This project is for academic purposes (CSC 426 — Deep Learning, Spring 2026).

Dataset: [euclaise/writingprompts](https://huggingface.co/datasets/euclaise/writingprompts) (Hugging Face, open access).
