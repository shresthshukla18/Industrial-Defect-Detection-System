# Cell 8 

%%writefile app.py
import streamlit as st
import os
import pandas as pd
from PIL import Image
import time

# ==========================================
#  CONFIGURATION
# ==========================================
st.set_page_config(page_title="PoC-5 Dashboard", layout="wide")
st.title("🏭 Defect Surveillance Dashboard")

PROJECT_PATH = "/content/drive/MyDrive/POC5_Project"
OUTPUTS_PATH = os.path.join(PROJECT_PATH, "outputs")
RESULTS_PATH = os.path.join(PROJECT_PATH, "results")

# ==========================================
#  DATA GATHERING & PARSING
# ==========================================
if not os.path.exists(OUTPUTS_PATH) or not os.path.exists(os.path.join(OUTPUTS_PATH, "images")):
    st.error("No processed outputs found. Please run your pipeline first.")
    st.stop()

all_runs = []
image_base_dir = os.path.join(OUTPUTS_PATH, "images")
for category in os.listdir(image_base_dir):
    cat_dir = os.path.join(image_base_dir, category)
    if not os.path.isdir(cat_dir): continue

    for file in os.listdir(cat_dir):
        if file.endswith("_original.png"):
            file_path = os.path.join(cat_dir, file)
            mtime = os.path.getmtime(file_path)

            base_name = file.replace("_original.png", "")
            name_parts = base_name.split("_")

            if len(name_parts) > 1 and name_parts[-1].isdigit():
                defect_type = " ".join(name_parts[:-1]).title()
            elif len(name_parts) == 1 and name_parts[0].isdigit():
                defect_type = "Legacy Run (Unrecorded Defect)"
            else:
                defect_type = base_name.replace("_", " ").title()

            all_runs.append({
                "category": category.capitalize(),
                "category_raw": category,
                "defect": defect_type,
                "base_name": base_name,
                "mtime": mtime,
                "paths": {
                    "orig": file_path,
                    "box": os.path.join(OUTPUTS_PATH, "localization", category, f"{base_name}_ae_box.png"),
                    "padim": os.path.join(OUTPUTS_PATH, "overlays", category, f"{base_name}_padim_overlay.png"),
                    "patch": os.path.join(OUTPUTS_PATH, "overlays", category, f"{base_name}_patchcore_overlay.png")
                }
            })

if not all_runs:
    st.info("No images found in outputs.")
    st.stop()

all_runs.sort(key=lambda x: x["mtime"], reverse=True)

current_run = all_runs[0]
history_runs = all_runs[1:]

# ==========================================
#  UI LAYOUT (TABS)
# ==========================================
tab1, tab2 = st.tabs([" Current Run", " History Panel"])

# --- TAB 1: CURRENT RUN ---
with tab1:
    st.header(f"Target Category: {current_run['category']}")
    st.subheader(f"Detected Condition: {current_run['defect']}")

    cols = st.columns(4)
    paths = current_run["paths"]
    if os.path.exists(paths["orig"]): cols[0].image(Image.open(paths["orig"]), caption="Original Image", use_container_width=True)
    if os.path.exists(paths["box"]): cols[1].image(Image.open(paths["box"]), caption="Autoencoder", use_container_width=True)
    if os.path.exists(paths["padim"]): cols[2].image(Image.open(paths["padim"]), caption="PaDiM Overlay", use_container_width=True)
    if os.path.exists(paths["patch"]): cols[3].image(Image.open(paths["patch"]), caption="PatchCore Overlay", use_container_width=True)

    st.divider()

    st.markdown(f"### Analytics & Performance ({current_run['category']})")
    metrics_file = os.path.join(RESULTS_PATH, f"{current_run['category_raw']}_evaluation.csv")

    if os.path.exists(metrics_file):
        st.dataframe(pd.read_csv(metrics_file), use_container_width=True, hide_index=True)
        col_c1, col_c2 = st.columns([1, 2])

        auroc_img = os.path.join(RESULTS_PATH, f"{current_run['category_raw']}_auroc_graph.png")
        if os.path.exists(auroc_img):
            col_c1.image(Image.open(auroc_img), use_container_width=True)

        dist_cols = col_c2.columns(3)
        for col, model in zip(dist_cols, ["autoencoder", "padim", "patchcore"]):
            dist_img = os.path.join(RESULTS_PATH, f"{current_run['category_raw']}_{model}_score_distribution.png")
            if os.path.exists(dist_img):
                col.image(Image.open(dist_img), caption=model.capitalize(), use_container_width=True)
    else:
        st.warning("Analytics not generated yet. Run Cell 5 to populate this section.")

# --- TAB 2: HISTORY PANEL ---
with tab2:
    st.header("Previous Runs")
    if not history_runs:
        st.info("No previous runs found. Run your pipeline again to start building history.")
    else:
        for run in history_runs:
            run_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run['mtime']))
            with st.expander(f"🕰️ {run['category']} - {run['defect']} (Processed: {run_time})"):

                # 1. Show the images
                h_cols = st.columns(4)
                h_paths = run["paths"]
                if os.path.exists(h_paths["orig"]): h_cols[0].image(Image.open(h_paths["orig"]), caption="Original Image", use_container_width=True)
                if os.path.exists(h_paths["box"]): h_cols[1].image(Image.open(h_paths["box"]), caption="Autoencoder", use_container_width=True)
                if os.path.exists(h_paths["padim"]): h_cols[2].image(Image.open(h_paths["padim"]), caption="PaDiM Overlay", use_container_width=True)
                if os.path.exists(h_paths["patch"]): h_cols[3].image(Image.open(h_paths["patch"]), caption="PatchCore Overlay", use_container_width=True)

                st.divider()

                # 2. Show the analytics for that run's category
                st.markdown(f"#### 📊 Analytics ({run['category']})")
                hist_metrics_file = os.path.join(RESULTS_PATH, f"{run['category_raw']}_evaluation.csv")

                if os.path.exists(hist_metrics_file):
                    st.dataframe(pd.read_csv(hist_metrics_file), use_container_width=True, hide_index=True)
                    col_h1, col_h2 = st.columns([1, 2])

                    hist_auroc_img = os.path.join(RESULTS_PATH, f"{run['category_raw']}_auroc_graph.png")
                    if os.path.exists(hist_auroc_img):
                        col_h1.image(Image.open(hist_auroc_img), use_container_width=True)

                    hist_dist_cols = col_h2.columns(3)
                    for col, model in zip(hist_dist_cols, ["autoencoder", "padim", "patchcore"]):
                        hist_dist_img = os.path.join(RESULTS_PATH, f"{run['category_raw']}_{model}_score_distribution.png")
                        if os.path.exists(hist_dist_img):
                            col.image(Image.open(hist_dist_img), caption=model.capitalize(), use_container_width=True)
                else:
                    st.info("Analytics not found for this historical run.")
