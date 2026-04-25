#CELL1

# ==============================
# COLAB SETUP (ROBUST)
# ==============================

import os
import shutil
import torch

# ===== 1. VERIFY COLAB =====
try:
    import google.colab
    print(" Running in Google Colab")
except:
    raise Exception(" Not running in Colab. Open notebook in Google Colab.")

# ===== 2. GPU CHECK =====
print("Torch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print(" WARNING: No GPU. Go to Runtime > Change runtime > GPU")

# ===== 3. INSTALL DEPENDENCIES =====
!pip install -q anomalib timm torchmetrics opencv-python kagglehub

print(" Dependencies installed")

# ===== 4. CLEAN REMOUNT (FIXES YOUR ERROR) =====
from google.colab import drive

try:
    drive.mount('/content/drive', force_remount=True)
    print(" Drive mounted successfully")
except Exception as e:
    print(" Drive mount failed. Retrying...")
    drive.mount('/content/drive', force_remount=True)

# ===== 5. DEFINE PATHS =====
PROJECT_PATH = "/content/drive/MyDrive/POC5_Project"
DATASET_BASE = os.path.join(PROJECT_PATH, "dataset/mvtec-ad")

print(" Project path:", PROJECT_PATH)

# ===== 6. CREATE FOLDERS =====
folders = [
    "dataset/mvtec-ad",
    "models/autoencoder",
    "models/padim",
    "models/patchcore",
    "outputs/images",
    "outputs/heatmaps",
    "outputs/localization",
    "outputs/overlays",
    "results"
]

for folder in folders:
    os.makedirs(os.path.join(PROJECT_PATH, folder), exist_ok=True)

print(" Folder structure ready")

# ===== 7. DOWNLOAD DATASET (SAFE) =====
if os.path.exists(DATASET_BASE) and len(os.listdir(DATASET_BASE)) > 0:
    print(" Dataset already exists")

else:
    print("⬇ Downloading dataset...")

    import kagglehub
    path = kagglehub.dataset_download("ipythonx/mvtec-ad")

    print("Downloaded to:", path)

    shutil.copytree(path, DATASET_BASE, dirs_exist_ok=True)
    print(" Dataset copied to Drive")

print(" Dataset path:", DATASET_BASE)

# ===== 8. VERIFY DATASET =====
all_categories = [
    c for c in os.listdir(DATASET_BASE)
    if os.path.isdir(os.path.join(DATASET_BASE, c))
]

if len(all_categories) == 0:
    raise Exception(" Dataset is empty or corrupted")

print("\nTotal categories:", len(all_categories))
print(all_categories)

# ===== 9. SELECT CATEGORY =====
CATEGORY = all_categories[0]
print("\nUsing CATEGORY:", CATEGORY)

# ===== 10. VERIFY TRAIN DATA =====
train_path = os.path.join(DATASET_BASE, CATEGORY, "train", "good")

if not os.path.exists(train_path):
    raise Exception(" Train folder missing")

train_images = os.listdir(train_path)
print(f" Found {len(train_images)} training images")
