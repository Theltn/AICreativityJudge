# Google Colab Training — Setup Instructions

## What You Need
- A Google account with access to [Google Colab](https://colab.research.google.com/)
- The 3 data split files from your local machine:
  - `data/processed/train.parquet` (190,805 stories)
  - `data/processed/val.parquet` (40,887 stories)
  - `data/processed/test.parquet` (40,887 stories)

---

## Step 1: Upload Data to Google Drive

1. Open [Google Drive](https://drive.google.com/)
2. Create a folder called `AICreativityJudge`
3. Inside that folder, create a subfolder called `data`
4. Upload the 3 parquet files into that `data` folder

Your Drive structure should look like:
```
My Drive/
  AICreativityJudge/
    data/
      train.parquet
      val.parquet
      test.parquet
```

---

## Step 2: Run MLP Baseline (fast, ~1 minute)

1. Open [Google Colab](https://colab.research.google.com/)
2. Click `File → Upload notebook`
3. Upload `notebooks/02_MLP_Baseline.ipynb` from your local project
4. Click `Runtime → Run all`
5. When prompted, authorize Google Drive access
6. Wait for it to complete (~1 min)
7. Results will print at the bottom + saved to your Drive

---

## Step 3: Run RoBERTa Fine-tuning (GPU required, ~2-4 hours)

1. Open [Google Colab](https://colab.research.google.com/)
2. Click `File → Upload notebook`
3. Upload `notebooks/03_RoBERTa_Finetuning.ipynb` from your local project
4. **IMPORTANT:** Click `Runtime → Change runtime type → T4 GPU`
5. Click `Runtime → Run all`
6. When prompted, authorize Google Drive access
7. Monitor training progress (prints every 200 steps)
8. When done, the trained model is saved to your Drive at:
   `AICreativityJudge/data/roberta_creativity_model/`

---

## Step 4: Download Results Back to Local

After both notebooks finish, your Google Drive `data/` folder will contain:
- `mlp_best.pt` — MLP model weights
- `mlp_training_curves.png` — training plot
- `mlp_scatter.png` — predicted vs true scatter
- `roberta_creativity_model/` — full RoBERTa model (for inference)
- `roberta_training_curves.png` — training plot
- `roberta_scatter.png` — predicted vs true scatter
- `roberta_test_metrics.json` — final test MSE/MAE/R²

Download these into your local `data/models/` folder for the next phase.

---

## Troubleshooting

- **"Out of memory" on RoBERTa:** Reduce `BATCH_SIZE` from 16 to 8 in the training cell
- **Drive won't mount:** Make sure you click "Allow" on the popup
- **Slow training:** Verify GPU is active: `Runtime → Change runtime type` should show T4
- **Disconnected mid-training:** Re-run from the top — the model auto-saves the best checkpoint
