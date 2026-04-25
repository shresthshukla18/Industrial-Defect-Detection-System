#CELL3
import os
import random
import torch

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===== RANDOMIZE CATEGORY =====
# Find all category folders dynamically
all_categories = sorted([d for d in os.listdir(DATASET_BASE) if os.path.isdir(os.path.join(DATASET_BASE, d))])

if len(all_categories) == 0:
    raise Exception("No categories found in your dataset folder!")

# Pick a completely random category!
CATEGORY = random.choice(all_categories)
print("\n==============================")
print(" Randomly Selected CATEGORY:", CATEGORY.upper())
print("==============================")

test_path = os.path.join(DATASET_BASE, CATEGORY, "test")
if not os.path.exists(test_path):
    raise Exception(f"Test path not found: {test_path}")

# ===== GET DEFECT TYPES =====
# Find all folders that are not "good"
defect_types = [d for d in os.listdir(test_path) if d != "good"]
if len(defect_types) == 0:
    raise Exception(f"No defect types found for {CATEGORY}")

print("Available defects:", defect_types)

# ===== RANDOMIZE DEFECT & IMAGE =====
selected_defect = random.choice(defect_types)
defect_path = os.path.join(test_path, selected_defect)
test_images = sorted([f for f in os.listdir(defect_path) if f.endswith(('.png', '.jpg'))])

if len(test_images) == 0:
    raise Exception("No test images found")

img_name = random.choice(test_images)
test_img_path = os.path.join(defect_path, img_name)

print(" Selected defect:", selected_defect)
print(" Selected image:", img_name)
