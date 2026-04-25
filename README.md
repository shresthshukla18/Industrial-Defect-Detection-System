#  PoC-5: Industrial Defect Detection (Few-Shot Learning)

##  Project Objective
Build a robust anomaly detection system capable of identifying industrial defects with **minimal or no defective training samples**, using unsupervised and one-class learning approaches.

---

##  Problem Definition
Industrial datasets are **highly imbalanced**, where defective samples are rare.  
Traditional supervised models fail due to lack of anomaly data.

This project addresses:
- Detection of unseen defects
- Pixel-level localization
- Reliable scoring under few-shot conditions

---

##  System Architecture
Pipeline consists of:

    1. Training (Offline)
       - Autoencoder → reconstruction learning
       - PaDiM → statistical feature modeling
       - PatchCore → memory bank (coreset sampling)

    2. Inference
       - Random defect image selection
       - Anomaly map generation (pixel-level)

    3. Post-processing
       - Heatmaps
       - Bounding box localization

    4. Visualization Layer (UI)
       - Streamlit dashboard
       - History tracking + analytics display

---

##  Key Features
- One-class anomaly detection (no defect training required)
- Multi-model comparison (Autoencoder, PaDiM, PatchCore)
- Pixel-level defect localization
- Heatmaps + overlay visualization
- Automated evaluation pipeline
- Interactive Streamlit dashboard with history panel
- Cloud deployment using Cloudflare tunnel

---

##  Project Structure

    POC5_Project/
    │
    ├── dataset/mvtec-ad/
    ├── models/
    │ ├── autoencoder/
    │ ├── padim/
    │ └── patchcore/
    │
    ├── outputs/
    │ ├── images/
    │ ├── heatmaps/
    │ ├── localization/
    │ └── overlays/
    │
    ├── results/
    │ ├── evaluation.csv
    │ ├── graphs
    │ └── score_distributions
    │
    └── app.py (Streamlit UI)

    
---

##  Installation
    ```bash
    pip install anomalib timm torchmetrics opencv-python kagglehub streamlit

---

## How to Run

How to Run 

    Cell 1 → Setup + dataset
    Cell 2 → Train models (all categories)
    Cell 3 → Random test selection
    Cell 5 → Inference + output generation
    Cell 6 → Evaluation + metrics
    Cell 8 → Streamlit UI
    Cell 9 → Cloudflare deployment 

---

## Outputs Generated

Stored under /outputs/:

/images/ → original images

/heatmaps/ → AE, PaDiM, PatchCore maps

/localization/ → AE bounding boxes

/overlays/ → PaDiM & PatchCore overlays

---

## Key Metrics

    Image-level ROC-AUC
    Pixel-level ROC-AUC
    Image-level F1 Score
    Pixel-level F1 Score
    PR-AUC (Autoencoder only)
    Anomaly score distributions

---

## Limitations

* No real-time UI-driven inference (batch-only system)
* PaDiM & PatchCore require full dataset context (not lightweight)
* Autoencoder capped at 30 epochs (no deep tuning)
* No Precision-Recall curve visualization (only PR-AUC value)
* No inference speed benchmarking (FPS not measured)
* No model size or memory optimization
* No formal false positive statistical analysis
* UI depends entirely on pre-generated outputs

---

## Observed Failure Scenarios

* Low-contrast defects → weak anomaly map
* Texture-heavy categories → false positives
* Autoencoder reconstructs minor defects
* PatchCore sensitive to feature noise
* Boundary localization may be imprecise

---

## Scalability

* Supports multiple categories dynamically
* Modular model training pipeline
* UI supports multi-run tracking via filesystem

---

## Future Improvements

* Real-time inference pipeline
* Model optimization (quantization / TensorRT)
* Precision-Recall curve visualization
* False positive reduction techniques
* Explainability enhancements

---

## Conclusion

This PoC implements a fully functional industrial anomaly detection pipeline using one-class learning.

It demonstrates:

Effective defect detection without labeled anomalies
Comparative performance across reconstruction and embedding methods
Practical visualization and evaluation workflow

The system prioritizes accuracy, modularity, and reproducibility over real-time deployment constraints.
