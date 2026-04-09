"""
Local Training Script — MLP + RoBERTa
=======================================
Designed for local machines with NVIDIA GPUs (CUDA).
Run: python scripts/train_local.py

Trains both models sequentially:
  1. MLP Baseline (~1 min)
  2. RoBERTa Fine-tuning (~1-4 hrs depending on GPU)
"""

import os
import sys
import time
import json
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────
DATA_DIR = "data/processed"
OUTPUT_DIR = "data/models"
FEATURE_COLS = [
    "lexical_richness", "syntactic_complexity",
    "novelty", "imagery", "narrative_dynamics"
]
TARGET_COL = "composite_score"
TEXT_COL = "story_truncated"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Device Setup ───────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS")
else:
    device = torch.device("cpu")
    print("Using CPU (this will be slow for RoBERTa)")


# ══════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  LOADING DATA")
print("=" * 60)

train_df = pd.read_parquet(f"{DATA_DIR}/train.parquet")
val_df = pd.read_parquet(f"{DATA_DIR}/val.parquet")
test_df = pd.read_parquet(f"{DATA_DIR}/test.parquet")

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")


# ══════════════════════════════════════════════════════════════
#  PART 1: MLP BASELINE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PART 1: MLP BASELINE")
print("=" * 60)


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


def df_to_tensors(df):
    X = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
    y = torch.tensor(df[TARGET_COL].values, dtype=torch.float32).unsqueeze(1)
    return X, y


X_train, y_train = df_to_tensors(train_df)
X_val, y_val = df_to_tensors(val_df)
X_test, y_test = df_to_tensors(test_df)

mlp_model = CreativityMLP().to(device)
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

BATCH_SIZE = 256
train_loader_mlp = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

mlp_train_losses = []
mlp_val_losses = []
best_mlp_val = float("inf")

print("Training MLP...")
for epoch in range(50):
    mlp_model.train()
    epoch_loss = 0
    for xb, yb in train_loader_mlp:
        xb, yb = xb.to(device), yb.to(device)
        pred = mlp_model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    train_loss = epoch_loss / len(X_train)
    mlp_train_losses.append(train_loss)

    mlp_model.eval()
    with torch.no_grad():
        val_pred = mlp_model(X_val.to(device))
        val_loss = criterion(val_pred, y_val.to(device)).item()
    mlp_val_losses.append(val_loss)
    scheduler.step(val_loss)

    if val_loss < best_mlp_val:
        best_mlp_val = val_loss
        torch.save(mlp_model.state_dict(), f"{OUTPUT_DIR}/mlp_best.pt")

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/50  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}")

# MLP Evaluation
mlp_model.load_state_dict(torch.load(f"{OUTPUT_DIR}/mlp_best.pt"))
mlp_model.eval()
with torch.no_grad():
    mlp_preds = mlp_model(X_test.to(device)).cpu().numpy().flatten()
    mlp_true = y_test.numpy().flatten()

mlp_mse = mean_squared_error(mlp_true, mlp_preds)
mlp_mae = mean_absolute_error(mlp_true, mlp_preds)
mlp_r2 = r2_score(mlp_true, mlp_preds)

print(f"\n  MLP TEST RESULTS:")
print(f"  MSE={mlp_mse:.4f}  RMSE={np.sqrt(mlp_mse):.4f}  MAE={mlp_mae:.4f}  R²={mlp_r2:.4f}")

# Save MLP plots
plt.figure(figsize=(10, 5))
plt.plot(mlp_train_losses, label="Train", color="#00d287")
plt.plot(mlp_val_losses, label="Val", color="#00e5ff")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("MLP Training Curves")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mlp_training_curves.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 8))
plt.scatter(mlp_true, mlp_preds, alpha=0.05, s=5, color="#00d287")
plt.plot([0, 10], [0, 10], "r--", alpha=0.7)
plt.xlabel("True"); plt.ylabel("Predicted"); plt.title(f"MLP (R²={mlp_r2:.3f})")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mlp_scatter.png", dpi=150)
plt.close()

json.dump({"mse": mlp_mse, "mae": mlp_mae, "r2": mlp_r2},
          open(f"{OUTPUT_DIR}/mlp_metrics.json", "w"), indent=2)

print("  MLP complete. Plots saved.\n")


# ══════════════════════════════════════════════════════════════
#  PART 2: RoBERTa FINE-TUNING
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("  PART 2: RoBERTa FINE-TUNING")
print("=" * 60)

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)


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
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.scores[idx], dtype=torch.float32),
        }


print("Loading tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

train_dataset = StoryDataset(train_df[TEXT_COL].tolist(), train_df[TARGET_COL].tolist(), tokenizer)
val_dataset = StoryDataset(val_df[TEXT_COL].tolist(), val_df[TARGET_COL].tolist(), tokenizer)
test_dataset = StoryDataset(test_df[TEXT_COL].tolist(), test_df[TARGET_COL].tolist(), tokenizer)

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

print(f"Batches per epoch: {len(train_loader)}")

print("Loading RoBERTa-base...")
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=1, problem_type="regression"
)
model = model.to(device)

EPOCHS = 3
LR = 2e-5

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * 0.1)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

print(f"Total steps: {total_steps}  Warmup: {warmup_steps}")
print(f"Training for {EPOCHS} epochs...\n")

train_losses = []
val_losses = []
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

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
            rate = (step + 1) / elapsed
            eta = (len(train_loader) - step - 1) / rate / 60
            print(f"  Epoch {epoch+1} step {step+1}/{len(train_loader)}  "
                  f"loss={loss.item():.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"ETA: {eta:.0f}min")

    avg_train = epoch_loss / len(train_loader)
    train_losses.append(avg_train)

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
    avg_val = val_loss / len(val_loader)
    val_losses.append(avg_val)

    elapsed = time.time() - t0
    print(f"\n  Epoch {epoch+1}/{EPOCHS}  train={avg_train:.4f}  val={avg_val:.4f}  time={elapsed/60:.1f}min")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/roberta_best.pt")
        print(f"  → Saved best model")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── Test Evaluation ────────────────────────────────────────────
print(f"\nEvaluating on test set...")
model.load_state_dict(torch.load(f"{OUTPUT_DIR}/roberta_best.pt"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_preds.extend(outputs.logits.squeeze(-1).cpu().numpy())
        all_labels.extend(batch["labels"].numpy())

y_true = np.array(all_labels)
y_pred = np.array(all_preds)

rob_mse = mean_squared_error(y_true, y_pred)
rob_mae = mean_absolute_error(y_true, y_pred)
rob_r2 = r2_score(y_true, y_pred)

print(f"\n  RoBERTa TEST RESULTS:")
print(f"  MSE={rob_mse:.4f}  RMSE={np.sqrt(rob_mse):.4f}  MAE={rob_mae:.4f}  R²={rob_r2:.4f}")

# Save model + plots
model.save_pretrained(f"{OUTPUT_DIR}/roberta_creativity_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/roberta_creativity_model")

plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS+1), train_losses, "o-", label="Train", color="#00d287")
plt.plot(range(1, EPOCHS+1), val_losses, "o-", label="Val", color="#00e5ff")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("RoBERTa Training Curves")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roberta_training_curves.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.05, s=5, color="#00e5ff")
plt.plot([0, 10], [0, 10], "r--", alpha=0.7)
plt.xlabel("True"); plt.ylabel("Predicted"); plt.title(f"RoBERTa (R²={rob_r2:.3f})")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roberta_scatter.png", dpi=150)
plt.close()

json.dump({"mse": rob_mse, "mae": rob_mae, "r2": rob_r2},
          open(f"{OUTPUT_DIR}/roberta_metrics.json", "w"), indent=2)


# ══════════════════════════════════════════════════════════════
#  COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  MODEL COMPARISON")
print("=" * 60)
print(f"  {'Metric':<8} {'MLP':>10} {'RoBERTa':>10}")
print(f"  {'MSE':<8} {mlp_mse:>10.4f} {rob_mse:>10.4f}")
print(f"  {'RMSE':<8} {np.sqrt(mlp_mse):>10.4f} {np.sqrt(rob_mse):>10.4f}")
print(f"  {'MAE':<8} {mlp_mae:>10.4f} {rob_mae:>10.4f}")
print(f"  {'R²':<8} {mlp_r2:>10.4f} {rob_r2:>10.4f}")
print("=" * 60)
print("\n✅ All training complete! Models saved to data/models/")
