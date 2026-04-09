"""
Generate Training Notebooks for Google Colab
=============================================
Creates two notebooks:
  1. MLP Baseline (fast, ~1 min)
  2. RoBERTa Fine-tuning (GPU required, ~2-4 hrs)
"""

import json
import os


def make_cell(cell_type, source, metadata=None):
    """Create a notebook cell."""
    return {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": [line + "\n" for line in source.rstrip("\n").split("\n")],
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def create_mlp_notebook():
    cells = []

    # Title
    cells.append(make_cell("markdown", """# MLP Baseline — Creativity Score Prediction
---
This notebook trains a simple Multi-Layer Perceptron on the **5 rubric features** to predict the composite creativity score.

**Architecture:** `5 features → 64 → 32 → 1 (score)`

⚠️ **Before running:** Upload `train.parquet`, `val.parquet`, and `test.parquet` to your Google Drive under a folder called `AICreativityJudge/data/`.
"""))

    # Setup
    cells.append(make_cell("code", """# ── Install & Imports ──
!pip install -q pandas pyarrow scikit-learn torch

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = '/content/drive/MyDrive/AICreativityJudge/data'
print("Setup complete.")
"""))

    # Load Data
    cells.append(make_cell("code", """# ── Load Splits ──
train_df = pd.read_parquet(f'{DATA_DIR}/train.parquet')
val_df   = pd.read_parquet(f'{DATA_DIR}/val.parquet')
test_df  = pd.read_parquet(f'{DATA_DIR}/test.parquet')

FEATURE_COLS = [
    'lexical_richness', 'syntactic_complexity',
    'novelty', 'imagery', 'narrative_dynamics'
]
TARGET_COL = 'composite_score'

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
print(f"\\nFeature stats (train):")
print(train_df[FEATURE_COLS].describe().round(2))
"""))

    # Prepare tensors
    cells.append(make_cell("code", """# ── Prepare PyTorch Tensors ──
def df_to_tensors(df):
    X = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
    y = torch.tensor(df[TARGET_COL].values, dtype=torch.float32).unsqueeze(1)
    return X, y

X_train, y_train = df_to_tensors(train_df)
X_val, y_val     = df_to_tensors(val_df)
X_test, y_test   = df_to_tensors(test_df)

BATCH_SIZE = 256
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

print(f"Input shape: {X_train.shape}  Target shape: {y_train.shape}")
"""))

    # Model Definition
    cells.append(make_cell("code", """# ── MLP Model ──
class CreativityMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CreativityMLP().to(device)
print(f"Device: {device}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(model)
"""))

    # Training
    cells.append(make_cell("code", """# ── Train ──
EPOCHS = 50
LR = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    # Train
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    train_loss = epoch_loss / len(X_train)
    train_losses.append(train_loss)

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += criterion(pred, yb).item() * len(xb)
    val_loss /= len(X_val)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'{DATA_DIR}/mlp_best.pt')

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")

print(f"\\nBest val MSE: {best_val_loss:.4f}")
"""))

    # Training Curves
    cells.append(make_cell("code", """# ── Training Curves ──
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train MSE', color='#00d287')
plt.plot(val_losses, label='Val MSE', color='#00e5ff')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MLP Training Curves')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/mlp_training_curves.png', dpi=150)
plt.show()
"""))

    # Evaluation
    cells.append(make_cell("code", """# ── Evaluate on Test Set ──
model.load_state_dict(torch.load(f'{DATA_DIR}/mlp_best.pt'))
model.eval()

with torch.no_grad():
    y_pred = model(X_test.to(device)).cpu().numpy().flatten()
    y_true = y_test.numpy().flatten()

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2  = r2_score(y_true, y_pred)

print("=" * 50)
print("  MLP BASELINE — TEST SET RESULTS")
print("=" * 50)
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {np.sqrt(mse):.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r2:.4f}")
print("=" * 50)

# Scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.05, s=5, color='#00d287')
plt.plot([0, 10], [0, 10], 'r--', alpha=0.7, label='Perfect')
plt.xlabel('True Composite Score')
plt.ylabel('Predicted Composite Score')
plt.title(f'MLP Baseline: Predicted vs True (R²={r2:.3f})')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/mlp_scatter.png', dpi=150)
plt.show()
"""))

    return cells


def create_roberta_notebook():
    cells = []

    # Title
    cells.append(make_cell("markdown", """# RoBERTa Fine-tuning — Creativity Score Prediction
---
This notebook fine-tunes **RoBERTa-base** to predict creativity scores from raw story text.

**Architecture:** `[Story Text] → RoBERTa Tokenizer → RoBERTa-base → [CLS] → Linear(768→1) → Score (0–10)`

⚙️ **Runtime:** Set to **GPU (T4)** via `Runtime → Change runtime type → T4 GPU`

⚠️ **Before running:** Upload `train.parquet`, `val.parquet`, and `test.parquet` to your Google Drive under a folder called `AICreativityJudge/data/`.

⏱️ **Expected time:** ~2-4 hours on a T4 GPU
"""))

    # Setup
    cells.append(make_cell("code", """# ── Install & Imports ──
!pip install -q transformers datasets accelerate pandas pyarrow scikit-learn

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time, os, gc

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = '/content/drive/MyDrive/AICreativityJudge/data'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""))

    # Load Data
    cells.append(make_cell("code", """# ── Load Data ──
train_df = pd.read_parquet(f'{DATA_DIR}/train.parquet')
val_df   = pd.read_parquet(f'{DATA_DIR}/val.parquet')
test_df  = pd.read_parquet(f'{DATA_DIR}/test.parquet')

TEXT_COL = 'story_truncated'
TARGET_COL = 'composite_score'

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
print(f"\\nComposite score stats (train):")
print(train_df[TARGET_COL].describe().round(2))
"""))

    # Dataset class
    cells.append(make_cell("code", """# ── PyTorch Dataset ──
class StoryDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_length=512):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.scores[idx], dtype=torch.float32)
        }

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

train_dataset = StoryDataset(
    train_df[TEXT_COL].tolist(),
    train_df[TARGET_COL].tolist(),
    tokenizer
)
val_dataset = StoryDataset(
    val_df[TEXT_COL].tolist(),
    val_df[TARGET_COL].tolist(),
    tokenizer
)
test_dataset = StoryDataset(
    test_df[TEXT_COL].tolist(),
    test_df[TARGET_COL].tolist(),
    tokenizer
)

BATCH_SIZE = 16  # T4 can handle 16 with RoBERTa-base
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2)

print(f"Batches per epoch: {len(train_loader)}")
print("Datasets ready.")
"""))

    # Model setup
    cells.append(make_cell("code", """# ── Load RoBERTa for Regression ──
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=1,           # regression (single continuous output)
    problem_type='regression'
)
model = model.to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")
"""))

    # Training config
    cells.append(make_cell("code", """# ── Training Config ──
EPOCHS = 3
LR = 2e-5
WARMUP_RATIO = 0.1

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

print(f"Total training steps: {total_steps}")
print(f"Warmup steps: {warmup_steps}")
"""))

    # Training loop
    cells.append(make_cell("code", """# ── Training Loop ──
train_losses = []
val_losses = []
best_val_loss = float('inf')
save_path = f'{DATA_DIR}/roberta_best.pt'

for epoch in range(EPOCHS):
    # ── Train ──
    model.train()
    epoch_loss = 0
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

        if (step + 1) % 200 == 0:
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed
            eta_min = (len(train_loader) - step - 1) / steps_per_sec / 60
            print(f"  Epoch {epoch+1} step {step+1}/{len(train_loader)}  "
                  f"loss={loss.item():.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"ETA: {eta_min:.0f}min")

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ── Validate ──
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    elapsed = time.time() - t0
    print(f"\\nEpoch {epoch+1}/{EPOCHS}  "
          f"train_loss={avg_train_loss:.4f}  "
          f"val_loss={avg_val_loss:.4f}  "
          f"time={elapsed/60:.1f}min")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"  → Saved best model (val_loss={best_val_loss:.4f})")

    # Free up memory
    gc.collect()
    torch.cuda.empty_cache()

print(f"\\nTraining complete. Best val loss: {best_val_loss:.4f}")
"""))

    # Training curves
    cells.append(make_cell("code", """# ── Training Curves ──
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS+1), train_losses, 'o-', label='Train Loss', color='#00d287')
plt.plot(range(1, EPOCHS+1), val_losses, 'o-', label='Val Loss', color='#00e5ff')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('RoBERTa Training Curves')
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(range(1, EPOCHS+1))
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/roberta_training_curves.png', dpi=150)
plt.show()
"""))

    # Test evaluation
    cells.append(make_cell("code", """# ── Evaluate on Test Set ──
model.load_state_dict(torch.load(save_path))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.squeeze(-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

y_true = np.array(all_labels)
y_pred = np.array(all_preds)

mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

print("=" * 50)
print("  RoBERTa — TEST SET RESULTS")
print("=" * 50)
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r2:.4f}")
print("=" * 50)

# Scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.05, s=5, color='#00e5ff')
plt.plot([0, 10], [0, 10], 'r--', alpha=0.7, label='Perfect')
plt.xlabel('True Composite Score')
plt.ylabel('Predicted Composite Score')
plt.title(f'RoBERTa: Predicted vs True (R²={r2:.3f})')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/roberta_scatter.png', dpi=150)
plt.show()
"""))

    # Save full model
    cells.append(make_cell("code", """# ── Save Model for Inference ──
model.save_pretrained(f'{DATA_DIR}/roberta_creativity_model')
tokenizer.save_pretrained(f'{DATA_DIR}/roberta_creativity_model')

# Also save the test metrics
import json
metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
with open(f'{DATA_DIR}/roberta_test_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Model saved to {DATA_DIR}/roberta_creativity_model/")
print(f"Metrics saved to {DATA_DIR}/roberta_test_metrics.json")
print("\\n✅ Done! Download the model folder to your local project.")
"""))

    return cells


def save_notebook(cells, path):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {"name": "python", "version": "3.10.0"},
            "accelerator": "GPU",
            "gpuClass": "standard"
        },
        "cells": cells,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Notebook created at '{path}'")


if __name__ == "__main__":
    save_notebook(
        create_mlp_notebook(),
        "notebooks/02_MLP_Baseline.ipynb"
    )
    save_notebook(
        create_roberta_notebook(),
        "notebooks/03_RoBERTa_Finetuning.ipynb"
    )
