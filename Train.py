#CELL2

import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

os.environ["TQDM_DISABLE"] = "1"

# ===== ANOMALIB =====
from anomalib.models import Padim, Patchcore
from anomalib.engine import Engine
from anomalib.data import MVTecAD

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== CONTROL SWITCH =====
RETRAIN = False   #  Set True only when you WANT retraining


# ==============================
# AUTOENCODER
# ==============================
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ==============================
# DATASET
# ==============================
class MVTecTrainDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg'))]

        self.images = []
        for f in files:
            path = os.path.join(root_dir, f)
            if cv2.imread(path) is not None:
                self.images.append(f)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.images[idx])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img)


# ==============================
# GET CATEGORIES
# ==============================
all_categories = [
    c for c in os.listdir(DATASET_BASE)
    if os.path.isdir(os.path.join(DATASET_BASE, c))
]

print("Total categories:", len(all_categories))


# ==============================
# MAIN LOOP
# ==============================
for CATEGORY in all_categories:

    print("\n==============================")
    print("Processing:", CATEGORY)
    print("==============================")

    train_path = f"{DATASET_BASE}/{CATEGORY}/train/good"

    if not os.path.exists(train_path):
        print("Skipping:", CATEGORY)
        continue

    # =========================
    # AUTOENCODER
    # =========================
    ae_path = f"{PROJECT_PATH}/models/autoencoder/{CATEGORY}.pth"

    if os.path.exists(ae_path) and not RETRAIN:
        print(f"{CATEGORY} AE exists → skipping")
    else:
        print(f"{CATEGORY} AE training...")

        dataset = MVTecTrainDataset(train_path)
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

        ae_model = Autoencoder().to(device)
        optimizer = optim.Adam(ae_model.parameters(), lr=0.001)

        for epoch in range(30):
            total_loss = 0

            for imgs in loader:
                imgs = imgs.to(device)

                outputs = ae_model(imgs)
                loss = F.mse_loss(outputs, imgs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"{CATEGORY} AE Epoch {epoch+1}: {total_loss/len(loader):.6f}")

        os.makedirs(os.path.dirname(ae_path), exist_ok=True)
        torch.save(ae_model.state_dict(), ae_path)
        print(f"{CATEGORY} AE saved")


    # =========================
    # PADIM
    # =========================
    padim_path = f"{PROJECT_PATH}/models/padim/{CATEGORY}.ckpt"

    if os.path.exists(padim_path) and not RETRAIN:
        print(f"{CATEGORY} PaDiM exists → skipping")
    else:
        print(f"{CATEGORY} PaDiM training...")

        datamodule = MVTecAD(
            root=DATASET_BASE,
            category=CATEGORY,
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=2
        )
        datamodule.setup()

        padim_model = Padim(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"]
        )

        padim_engine = Engine(
            max_epochs=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1
        )

        padim_engine.fit(model=padim_model, datamodule=datamodule)

        padim_ckpt = padim_engine.trainer.checkpoint_callback.best_model_path

        if padim_ckpt:
            os.makedirs(os.path.dirname(padim_path), exist_ok=True)
            shutil.copy(padim_ckpt, padim_path)
            print(f"{CATEGORY} PaDiM saved")


    # =========================
    # PATCHCORE
    # =========================
    patch_path = f"{PROJECT_PATH}/models/patchcore/{CATEGORY}.ckpt"

    if os.path.exists(patch_path) and not RETRAIN:
        print(f"{CATEGORY} PatchCore exists → skipping")
    else:
        print(f"{CATEGORY} PatchCore training...")

        try:
            datamodule = MVTecAD(
                root=DATASET_BASE,
                category=CATEGORY,
                train_batch_size=32,
                eval_batch_size=32,
                num_workers=2
            )
            datamodule.setup()

            patch_model = Patchcore(
                backbone="resnet18",
                layers=["layer2", "layer3"],
                coreset_sampling_ratio=0.01
            )

            patch_engine = Engine(
                max_epochs=1,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                enable_progress_bar=False
            )

            patch_engine.fit(model=patch_model, datamodule=datamodule)

            patch_ckpt = patch_engine.trainer.checkpoint_callback.best_model_path

            if patch_ckpt:
                os.makedirs(os.path.dirname(patch_path), exist_ok=True)
                shutil.copy(patch_ckpt, patch_path)
                print(f"{CATEGORY} PatchCore saved")

        except Exception as e:
            print(f"{CATEGORY} PatchCore FAILED:", str(e))
