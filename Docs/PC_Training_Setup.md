# PC Setup — Local Training with NVIDIA GPU

## Quick Start (Windows with RTX 5070)

### 1. Clone the repo
```bash
git clone https://github.com/Theltn/AICreativityJudge.git
cd AICreativityJudge
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install PyTorch with CUDA
Go to https://pytorch.org/get-started/locally/ and get the right command for your setup.
For Windows + CUDA 12.x it's usually:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install other dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 5. Get the data files
You need these 3 files in `data/processed/`:
- `train.parquet`
- `val.parquet`
- `test.parquet`

**Transfer them from your Mac** — the easiest way:
- AirDrop them, USB drive, or upload to Google Drive and download on the PC
- They're in `/Users/theoh/Documents/Development/CSC426/AICreativityJudge/data/processed/`

### 6. Create the models output directory
```bash
mkdir data\models
```

### 7. Run training
```bash
python scripts/train_local.py
```

This will:
1. Train the MLP baseline (~1 min)
2. Fine-tune RoBERTa (~45 min – 1.5 hrs on RTX 5070)
3. Print a comparison of both models
4. Save everything to `data/models/`

### Verify GPU is being used
The script will print the GPU name at startup. You should see:
```
Using CUDA: NVIDIA GeForce RTX 5070
VRAM: 12.0 GB
```

If it says "Using CPU", PyTorch CUDA isn't installed correctly — re-do step 3.
