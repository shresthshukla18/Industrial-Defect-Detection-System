# Cell 6

# ==============================
# FINAL EVALUATION (ALL MODELS)
# ==============================
import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import transforms
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from anomalib.models import Padim, Patchcore
from anomalib.data import MVTecAD
from anomalib.engine import Engine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating category: {CATEGORY}")

# ==============================
# DATAMODULE
# ==============================
datamodule = MVTecAD(
    root=DATASET_BASE,
    category=CATEGORY,
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=2
)
datamodule.setup()

engine = Engine(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1
)

# ==============================
# PADIM
# ==============================
padim_ckpt = f"{PROJECT_PATH}/models/padim/{CATEGORY}.ckpt"
padim_model = Padim.load_from_checkpoint(padim_ckpt)

padim_results = engine.test(
    model=padim_model,
    datamodule=datamodule
)[0]

# ==============================
# PATCHCORE
# ==============================
patch_ckpt = f"{PROJECT_PATH}/models/patchcore/{CATEGORY}.ckpt"

patch_model = Patchcore.load_from_checkpoint(
    patch_ckpt,
    map_location=device,
    weights_only=False
)

patch_results = engine.test(
    model=patch_model,
    datamodule=datamodule
)[0]

# ==============================
# AUTOENCODER
# ==============================
ae_path = f"{PROJECT_PATH}/models/autoencoder/{CATEGORY}.pth"

ae_model = Autoencoder().to(device)
ae_model.load_state_dict(torch.load(ae_path, map_location=device))
ae_model.eval()

test_dir = os.path.join(DATASET_BASE, CATEGORY, "test")

y_true = []
y_scores = []

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

for defect_type in os.listdir(test_dir):
    defect_path = os.path.join(test_dir, defect_type)

    for img_name in os.listdir(defect_path):
        img_path = os.path.join(defect_path, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            recon = ae_model(input_tensor)

        recon_np = recon.cpu().squeeze().permute(1, 2, 0).numpy()
        input_np = input_tensor.cpu().squeeze().permute(1, 2, 0).numpy()

        error_map = np.abs(input_np - recon_np).mean(axis=2)
        score = error_map.mean()

        y_scores.append(score)
        y_true.append(0 if defect_type == "good" else 1)

# Metrics
ae_roc = roc_auc_score(y_true, y_scores)
precision, recall, _ = precision_recall_curve(y_true, y_scores)
ae_pr = auc(recall, precision)

# ==============================
# FINAL PRINT
# ==============================
print("\n===== FINAL RESULTS =====")

print("\n--- PaDiM ---")
print(padim_results)

print("\n--- PatchCore ---")
print(patch_results)

print("\n--- Autoencoder ---")
print({
    "image_AUROC": ae_roc,
    "PR_AUC": ae_pr
})

# ==============================
# SAVE RESULTS (CSV)
# ==============================
results_dir = os.path.join(PROJECT_PATH, "results")
os.makedirs(results_dir, exist_ok=True)

data = [
    {
        "Model": "Autoencoder",
        "Image_AUROC": float(ae_roc),
        "Pixel_AUROC": None,
        "Image_F1": None,
        "Pixel_F1": None,
        "PR_AUC": float(ae_pr)
    },
    {
        "Model": "PaDiM",
        "Image_AUROC": padim_results["image_AUROC"],
        "Pixel_AUROC": padim_results["pixel_AUROC"],
        "Image_F1": padim_results["image_F1Score"],
        "Pixel_F1": padim_results["pixel_F1Score"],
        "PR_AUC": None
    },
    {
        "Model": "PatchCore",
        "Image_AUROC": patch_results["image_AUROC"],
        "Pixel_AUROC": patch_results["pixel_AUROC"],
        "Image_F1": patch_results["image_F1Score"],
        "Pixel_F1": patch_results["pixel_F1Score"],
        "PR_AUC": None
    }
]

df = pd.DataFrame(data)

csv_path = os.path.join(results_dir, f"{CATEGORY}_evaluation.csv")
df.to_csv(csv_path, index=False)

print(f"✅ Results saved to: {csv_path}")

# ==============================
# DISPLAY TABLE
# ==============================
print("\n===== RESULT TABLE =====")
display(df)

# ==============================
# GRAPH (Image AUROC)
# ==============================
models = df["Model"]
image_auroc = df["Image_AUROC"]

plt.figure()
plt.bar(models, image_auroc)

for i, v in enumerate(image_auroc):
    if v is not None:
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')

plt.title("Model Comparison (Image AUROC)")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1.05)

graph_path = os.path.join(results_dir, f"{CATEGORY}_auroc_graph.png")
plt.savefig(graph_path, bbox_inches='tight')
plt.show()

print(f" Graph saved to: {graph_path}")

# ==============================
# ANOMALY SCORE DISTRIBUTION
# ==============================

# ---- PaDiM ----
padim_scores, padim_labels = [], []

padim_preds_for_plot = engine.predict(
    model=padim_model,
    datamodule=datamodule,
    ckpt_path=padim_ckpt
)

for p in padim_preds_for_plot:
    score = float(p["pred_score"][0].cpu().numpy().item())
    img_path = p.image_path[0]
    label = 0 if "good" in img_path else 1

    padim_scores.append(score)
    padim_labels.append(label)

good_scores_padim = [s for s, l in zip(padim_scores, padim_labels) if l == 0]
defect_scores_padim = [s for s, l in zip(padim_scores, padim_labels) if l == 1]

# ---- PatchCore ----
patch_scores, patch_labels = [], []

patch_preds_for_plot = engine.predict(
    model=patch_model,
    datamodule=datamodule,
    ckpt_path=patch_ckpt
)

for p in patch_preds_for_plot:
    score = float(p["pred_score"][0].cpu().numpy().item())
    img_path = p.image_path[0]
    label = 0 if "good" in img_path else 1

    patch_scores.append(score)
    patch_labels.append(label)

good_scores_patch = [s for s, l in zip(patch_scores, patch_labels) if l == 0]
defect_scores_patch = [s for s, l in zip(patch_scores, patch_labels) if l == 1]

# ---- Autoencoder ----
good_scores_ae = [s for s, l in zip(y_scores, y_true) if l == 0]
defect_scores_ae = [s for s, l in zip(y_scores, y_true) if l == 1]

# ==============================
# PLOTTING FUNCTION
# ==============================
def plot_and_save_scores(good_scores, defect_scores, model_name):
    plt.figure(figsize=(8, 6))

    plt.hist(good_scores, bins=30, alpha=0.6, label="Good")
    plt.hist(defect_scores, bins=30, alpha=0.6, label="Defect")

    plt.title(f"Anomaly Score Distribution ({model_name})")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.legend()

    save_path = os.path.join(
        results_dir,
        f"{CATEGORY}_{model_name.lower()}_score_distribution.png"
    )

    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    print(f" {model_name} Score plot saved to: {save_path}")
    plt.close()

# Run plots
plot_and_save_scores(good_scores_padim, defect_scores_padim, "PaDiM")
plot_and_save_scores(good_scores_patch, defect_scores_patch, "PatchCore")
plot_and_save_scores(good_scores_ae, defect_scores_ae, "Autoencoder")
