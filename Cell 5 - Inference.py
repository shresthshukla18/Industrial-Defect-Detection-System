# ==============================
# FINAL INFERENCE CELL (ENGINE BASED)
# ==============================
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from anomalib.models import Padim, Patchcore
from anomalib.engine import Engine
from anomalib.data import MVTecAD

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# LOAD IMAGE (FOR AE ONLY)
# ==============================
if not os.path.exists(test_img_path):
    raise Exception("Test image not found")

img = cv2.imread(test_img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

input_tensor = transform(img).unsqueeze(0).to(device)
input_np = input_tensor.cpu().squeeze().permute(1, 2, 0).numpy()

# ==============================
# AUTOENCODER
# ==============================
ae_path = f"{PROJECT_PATH}/models/autoencoder/{CATEGORY}.pth"

ae_model = Autoencoder().to(device)
ae_model.load_state_dict(torch.load(ae_path, map_location=device))
ae_model.eval()

with torch.no_grad():
    ae_recon = ae_model(input_tensor)

ae_recon_np = ae_recon.cpu().squeeze().permute(1, 2, 0).numpy()
ae_error = np.abs(input_np - ae_recon_np).mean(axis=2)

# AE localization
threshold = np.percentile(ae_error, 95)
mask = (ae_error > threshold).astype(np.uint8)

kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

mask = (mask * 255).astype(np.uint8)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

ae_box = (input_np * 255).astype(np.uint8).copy()

for cnt in contours:
    if cv2.contourArea(cnt) > 120:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(ae_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

# ==============================
# PADIM (ENGINE)
# ==============================
padim_ckpt = f"{PROJECT_PATH}/models/padim/{CATEGORY}.ckpt"

datamodule = MVTecAD(
    root=DATASET_BASE,
    category=CATEGORY,
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=0
)

datamodule.setup()

padim_model = Padim(
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3"]
)

padim_engine = Engine(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1
)

padim_preds = padim_engine.predict(
    model=padim_model,
    datamodule=datamodule,
    ckpt_path=padim_ckpt
)

padim_sample = padim_preds[0]
padim_img = padim_sample["image"][0]
padim_map = padim_sample["anomaly_map"][0]

padim_img = padim_img.cpu().numpy().transpose(1, 2, 0)
padim_map = padim_map.cpu().numpy()

# ==============================
# PATCHCORE (ENGINE)
# ==============================
patch_ckpt = f"{PROJECT_PATH}/models/patchcore/{CATEGORY}.ckpt"

patch_model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.01
)

patch_engine = Engine(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    enable_progress_bar=False
)

patch_preds = patch_engine.predict(
    model=patch_model,
    datamodule=datamodule,
    ckpt_path=patch_ckpt
)

patch_sample = patch_preds[0]
patch_img = patch_sample["image"][0]
patch_map = patch_sample["anomaly_map"][0]

patch_img = patch_img.cpu().numpy().transpose(1, 2, 0)
patch_map = patch_map.cpu().numpy()

# ==============================
# FINAL VISUALIZATION (RAW + OVERLAY)
# ==============================
def normalize_map(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def overlay_heatmap(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted((image * 255).astype(np.uint8), 0.75, heatmap, 0.25, 0)
    return overlay

# Normalize maps
ae_norm = normalize_map(ae_error)
padim_norm = normalize_map(padim_map)
patch_norm = normalize_map(patch_map)

# Overlays
padim_overlay = overlay_heatmap(input_np, padim_norm)
patch_overlay = overlay_heatmap(input_np, patch_norm)

# ==============================
# PLOTTING (3 ROWS)
# ==============================
plt.figure(figsize=(18, 14))

# Row 1 -> AE
plt.subplot(3, 3, 1)
plt.title("Original")
plt.imshow(input_np)
plt.axis("off")

plt.subplot(3, 3, 2)
plt.title("Reconstruction")
plt.imshow(ae_recon_np)
plt.axis("off")

plt.subplot(3, 3, 3)
plt.title("AE Raw Heatmap")
plt.imshow(ae_norm, cmap="jet")
plt.axis("off")

# Row 2 -> RAW MAPS
plt.subplot(3, 3, 4)
plt.title("AE Localization")
plt.imshow(ae_box)
plt.axis("off")

plt.subplot(3, 3, 5)
plt.title("PaDiM Raw Heatmap")
plt.imshow(padim_norm, cmap="jet")
plt.axis("off")

plt.subplot(3, 3, 6)
plt.title("PatchCore Raw Heatmap")
plt.imshow(patch_norm, cmap="jet")
plt.axis("off")

# Row 3 -> OVERLAYS
plt.subplot(3, 3, 7)
plt.title("AE (Reference)")
plt.imshow(input_np)
plt.axis("off")

plt.subplot(3, 3, 8)
plt.title("PaDiM Overlay")
plt.imshow(padim_overlay)
plt.axis("off")

plt.subplot(3, 3, 9)
plt.title("PatchCore Overlay")
plt.imshow(patch_overlay)
plt.axis("off")

plt.tight_layout()
plt.show()

# ==============================
# SAVE OUTPUTS (FIXED NAMING)
# ==============================
save_base = os.path.join(PROJECT_PATH, "outputs")

base_name = f"{selected_defect}_{os.path.basename(test_img_path).split('.')[0]}"

img_dir = os.path.join(save_base, "images", CATEGORY)
heatmap_dir = os.path.join(save_base, "heatmaps", CATEGORY)
loc_dir = os.path.join(save_base, "localization", CATEGORY)
overlay_dir = os.path.join(save_base, "overlays", CATEGORY)

os.makedirs(img_dir, exist_ok=True)
os.makedirs(heatmap_dir, exist_ok=True)
os.makedirs(loc_dir, exist_ok=True)
os.makedirs(overlay_dir, exist_ok=True)

# ORIGINAL
cv2.imwrite(
    os.path.join(img_dir, f"{base_name}_original.png"),
    cv2.cvtColor((input_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
)

# AE HEATMAP
ae_heatmap_color = cv2.applyColorMap(
    (ae_norm * 255).astype(np.uint8),
    cv2.COLORMAP_JET
)

cv2.imwrite(
    os.path.join(heatmap_dir, f"{base_name}_ae.png"),
    ae_heatmap_color
)

# PADIM HEATMAP
cv2.imwrite(
    os.path.join(heatmap_dir, f"{base_name}_padim.png"),
    (padim_norm * 255).astype(np.uint8)
)

# PATCHCORE HEATMAP
cv2.imwrite(
    os.path.join(heatmap_dir, f"{base_name}_patchcore.png"),
    (patch_norm * 255).astype(np.uint8)
)

# AE LOCALIZATION
cv2.imwrite(
    os.path.join(loc_dir, f"{base_name}_ae_box.png"),
    cv2.cvtColor(ae_box, cv2.COLOR_RGB2BGR)
)

# OVERLAYS
cv2.imwrite(
    os.path.join(overlay_dir, f"{base_name}_padim_overlay.png"),
    cv2.cvtColor(padim_overlay, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    os.path.join(overlay_dir, f"{base_name}_patchcore_overlay.png"),
    cv2.cvtColor(patch_overlay, cv2.COLOR_RGB2BGR)
)

print(f"Saved outputs for {CATEGORY} → {base_name}")
