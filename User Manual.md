#  PoC-5 User Manual

**Defect Detection using Autoencoder, PaDiM, and PatchCore**

---

## 1. System Requirements

* Google Colab (mandatory)
* GPU enabled (T4/V100 recommended)
* Google Drive storage (~5–10 GB free)

---

## 2. Execution Flow (STRICT ORDER)

You must run cells in exact sequence. No skipping.

---

###  Step 1: Environment Setup (CELL 1)

**Purpose:**

* Mount Google Drive
* Install dependencies
* Download MVTec AD dataset
* Create project folder structure

**Output:**

* Dataset loaded into Drive
* Categories detected
* Training images verified

---

###  Step 2: Model Training (CELL 2)

**Purpose:**

* Train all three models:

  * Autoencoder (30 epochs)
  * PaDiM (feature distribution)
  * PatchCore (memory bank)

**Important Controls:**

```python
RETRAIN = False
```

* `False` → skips already trained models
* `True` → retrains everything

**Output:**

* Saved models:

  ```
  models/
    autoencoder/
    padim/
    patchcore/
  ```

---

###  Step 3: Random Test Selection (CELL 3)

**Purpose:**

* Randomly selects:

  * Category
  * Defect type
  * Test image

**Output:**

* `CATEGORY`
* `selected_defect`
* `test_img_path`

---

###  Step 4: Cache Reset (CELL 4)

**Purpose:**

* Clears HuggingFace cache to avoid corrupted loads

```bash
!rm -rf ~/.cache/huggingface
```

---

###  Step 5: Inference + Visualization (CELL 5)

**Purpose:**

* Runs inference on:

  * Autoencoder
  * PaDiM
  * PatchCore

**Key Operations:**

* AE → reconstruction error
* PaDiM → Gaussian embedding anomaly
* PatchCore → memory bank distance

**Outputs Generated:**

1. **Visual Outputs**

   * Original image
   * Reconstruction
   * Heatmaps
   * Bounding boxes
   * Overlay maps

2. **Saved Files**

```
outputs/
 ├── images/
 ├── heatmaps/
 ├── localization/
 ├── overlays/
```

---

###  Step 6: Evaluation (CELL 6)

**Purpose:**

* Evaluate all models on full test dataset

**Metrics Computed:**

* Image AUROC
* Pixel AUROC (PaDiM, PatchCore)
* F1 Score
* PR-AUC (Autoencoder only)

**Outputs:**

1. **CSV File**

```
results/{CATEGORY}_evaluation.csv
```

2. **Graphs**

* AUROC comparison
* Score distribution plots

---

###  Step 7: UI Setup (CELL 7)

**Purpose:**

* Install Streamlit
* Install Cloudflare tunnel

---

###  Step 8: Dashboard (CELL 8)

**Purpose:**

* Create interactive UI

**Features:**

* Current run visualization
* Historical runs tracking
* Model comparison analytics
* Score distributions

---

###  Step 9: Launch UI (CELL 9)

**Purpose:**

* Start Streamlit server
* Expose via Cloudflare tunnel

**Output:**

* Public dashboard URL:

```
https://xxxxx.trycloudflare.com
```

---

## 3. Output Interpretation

###  Autoencoder

* Detects anomalies via reconstruction error
* Outputs:

  * Heatmap
  * Bounding box

###  PaDiM

* Uses feature distribution modeling
* Outputs:

  * Pixel-level anomaly map

###  PatchCore

* Uses memory bank similarity
* Outputs:

  * High-precision anomaly localization

---

## 4. File Structure 

```
POC5_Project/
│
├── dataset/mvtec-ad/
├── models/
│   ├── autoencoder/
│   ├── padim/
│   └── patchcore/
│
├── outputs/
│   ├── images/
│   ├── heatmaps/
│   ├── localization/
│   └── overlays/
│
├── results/
│   ├── evaluation.csv
│   ├── graphs
│   └── distributions
```

---

## 5. Critical Notes (Do NOT Ignore)

* Models must be trained before inference
* Cell 3 must run before Cell 5
* Cloudflare requires Streamlit to be active
* Dataset must remain inside Google Drive

---

## 6. Known Limitations

* No real-time UI inference
* No PR curve visualization (only PR-AUC)
* No speed benchmarking (FPS)
* No model size optimization

---

## 7. Recommended Usage Flow

1. Run Cell 1 → Setup
2. Run Cell 2 → Train
3. Run Cell 3 → Select test
4. Run Cell 5 → Inference
5. Run Cell 6 → Evaluation
6. Run Cell 7–9 → UI

---

## 8. End Result

* Fully working anomaly detection pipeline
* Visual + quantitative outputs
* Interactive dashboard with history

---
