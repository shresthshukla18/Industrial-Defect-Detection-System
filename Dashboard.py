%%writefile app.py
import streamlit as st
import os
import pandas as pd
from PIL import Image
import time
import io
import zipfile

# ==========================================
# 🔹 CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="PoC-5 Dashboard", page_icon="🏭", layout="wide")

# Inject Custom CSS for a professional, business-ready UI
st.markdown("""
<style>
    /* Main typography and colors */
    h1, h2, h3, h4 {
        color: #1E293B;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
    }
    
    /* Image styling: rounded corners, soft shadow, hover zoom */
    .stImage > img {
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.3s ease-in-out;
    }
    .stImage > img:hover {
        transform: scale(1.02);
    }
    
    /* Clean up the tabs to look like a modern dashboard */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F8FAFC;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 24px;
        border: 1px solid #E2E8F0;
        border-bottom: none;
        color: #64748B;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        color: #0F172A;
        border-top: 3px solid #2563EB;
        box-shadow: 0 -2px 4px rgba(0,0,0,0.02);
    }
    
    /* Style metrics and dataframes */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
    }
    
    /* Expander styling for history panel */
    .streamlit-expanderHeader {
        background-color: #F1F5F9;
        border-radius: 6px;
        font-weight: 500;
        color: #334155;
    }
    
    /* Custom divider */
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: 0;
        border-top: 1px solid #E2E8F0;
    }

    /* =========================================
       🟢 BUBBLE DOWNLOAD BUTTON
       ========================================= */
    .stDownloadButton > button {
        border-radius: 50px !important;
        background: linear-gradient(135deg, #00C6FF, #0072FF) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        animation: bubble-pulse 2s infinite !important;
        width: 100% !important; /* Makes it a nice wide pill shape */
    }
    .stDownloadButton > button:hover {
        transform: scale(1.02) translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 114, 255, 0.6) !important;
        animation: none !important;
    }
    .stDownloadButton > button:active {
        transform: scale(0.98) translateY(1px) !important;
        box-shadow: 0 2px 10px rgba(0, 114, 255, 0.3) !important;
    }
    @keyframes bubble-pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 114, 255, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(0, 114, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 114, 255, 0); }
    }
</style>
""", unsafe_allow_html=True)

# Dashboard Header
st.markdown("<h1 style='text-align: center; color: #0F172A; margin-bottom: 30px;'>🏭 Defect Surveillance Dashboard</h1>", unsafe_allow_html=True)

PROJECT_PATH = "/content/drive/MyDrive/POC5_Project"
OUTPUTS_PATH = os.path.join(PROJECT_PATH, "outputs")
RESULTS_PATH = os.path.join(PROJECT_PATH, "results")

# ==========================================
# 🔹 HELPER: ZIP GENERATOR
# ==========================================
def create_run_zip(run_data):
    """Bundles all images, CSVs, and graphs for a specific run into a single ZIP."""
    zip_buffer = io.BytesIO()
    cat_raw = run_data['category_raw']
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Add all visual evidence (Images & Heatmaps)
        for name, file_path in run_data['paths'].items():
            if os.path.exists(file_path):
                zip_file.write(file_path, arcname=f"evidence_images/{os.path.basename(file_path)}")
        
        # 2. Add Analytics Data (CSV)
        metrics_file = os.path.join(RESULTS_PATH, f"{cat_raw}_evaluation.csv")
        if os.path.exists(metrics_file):
            zip_file.write(metrics_file, arcname=f"analytics/{os.path.basename(metrics_file)}")
            
        # 3. Add Performance Graphs
        auroc_img = os.path.join(RESULTS_PATH, f"{cat_raw}_auroc_graph.png")
        if os.path.exists(auroc_img):
            zip_file.write(auroc_img, arcname=f"analytics/{os.path.basename(auroc_img)}")
            
        for model in ["autoencoder", "padim", "patchcore"]:
            dist_img = os.path.join(RESULTS_PATH, f"{cat_raw}_{model}_score_distribution.png")
            if os.path.exists(dist_img):
                zip_file.write(dist_img, arcname=f"analytics/{os.path.basename(dist_img)}")
                
    return zip_buffer.getvalue()

# ==========================================
# 🔹 DATA GATHERING & PARSING
# ==========================================
if not os.path.exists(OUTPUTS_PATH) or not os.path.exists(os.path.join(OUTPUTS_PATH, "images")):
    st.error("⚠️ No processed outputs found. Please run your pipeline first.")
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
                    "patch": os.path.join(OUTPUTS_PATH, "overlays", category, f"{base_name}_patchcore_overlay.png"),

                    # 🔥 ADDED (Cell 5 Heatmaps)
                    "ae_heat": os.path.join(OUTPUTS_PATH, "heatmaps", category, f"{base_name}_ae.png"),
                    "padim_heat": os.path.join(OUTPUTS_PATH, "heatmaps", category, f"{base_name}_padim.png"),
                    "patch_heat": os.path.join(OUTPUTS_PATH, "heatmaps", category, f"{base_name}_patchcore.png")
                }
            })

if not all_runs:
    st.info("ℹ️ No images found in outputs.")
    st.stop()

all_runs.sort(key=lambda x: x["mtime"], reverse=True)

current_run = all_runs[0]
history_runs = all_runs[1:]

# ==========================================
# 🔹 UI LAYOUT (TABS)
# ==========================================
tab1, tab2 = st.tabs(["🚀 Current Run Analysis", "📜 Historical Surveillance Panel"])

# --- TAB 1: CURRENT RUN ---
with tab1:
    st.markdown(f"<h2>Target Category: <span style='color: #2563EB;'>{current_run['category']}</span></h2>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color: #64748B; margin-bottom: 25px;'>Detected Condition: {current_run['defect']}</h4>", unsafe_allow_html=True)

    with st.container():
        cols = st.columns(4)
        paths = current_run["paths"]
        if os.path.exists(paths["orig"]): cols[0].image(Image.open(paths["orig"]), caption="Original Image", use_container_width=True)
        if os.path.exists(paths["box"]): cols[1].image(Image.open(paths["box"]), caption="Autoencoder", use_container_width=True)
        if os.path.exists(paths["padim"]): cols[2].image(Image.open(paths["padim"]), caption="PaDiM Overlay", use_container_width=True)
        if os.path.exists(paths["patch"]): cols[3].image(Image.open(paths["patch"]), caption="PatchCore Overlay", use_container_width=True)

    st.divider()

    # 🔥 ADDED HEATMAP SECTION
    with st.container():
        st.markdown("### 🌡️ Thermal / Defect Heatmaps")
        h_cols = st.columns(3)

        if os.path.exists(paths["ae_heat"]):
            h_cols[0].image(Image.open(paths["ae_heat"]), caption="AE Heatmap", use_container_width=True)

        if os.path.exists(paths["padim_heat"]):
            h_cols[1].image(Image.open(paths["padim_heat"]), caption="PaDiM Heatmap", use_container_width=True)

        if os.path.exists(paths["patch_heat"]):
            h_cols[2].image(Image.open(paths["patch_heat"]), caption="PatchCore Heatmap", use_container_width=True)

    st.divider()

    with st.container():
        st.markdown(f"### 📊 Analytics & Performance Metrics")
        metrics_file = os.path.join(RESULTS_PATH, f"{current_run['category_raw']}_evaluation.csv")

        if os.path.exists(metrics_file):
            st.dataframe(pd.read_csv(metrics_file), use_container_width=True, hide_index=True)
            
            st.markdown("<br>", unsafe_allow_html=True) # Spacer
            col_c1, col_c2 = st.columns([1, 2])

            auroc_img = os.path.join(RESULTS_PATH, f"{current_run['category_raw']}_auroc_graph.png")
            if os.path.exists(auroc_img):
                col_c1.image(Image.open(auroc_img), caption="AUROC Performance", use_container_width=True)

            dist_cols = col_c2.columns(3)
            for col, model in zip(dist_cols, ["autoencoder", "padim", "patchcore"]):
                dist_img = os.path.join(RESULTS_PATH, f"{current_run['category_raw']}_{model}_score_distribution.png")
                if os.path.exists(dist_img):
                    col.image(Image.open(dist_img), caption=f"{model.capitalize()} Distribution", use_container_width=True)
        else:
            st.warning("⚠️ Analytics not generated yet. Run Cell 5 to populate this section.")

    st.divider()
    
    # 🌟 CURRENT RUN DOWNLOAD BUTTON
    zip_data = create_run_zip(current_run)
    st.download_button(
        label="🫧 Download Full Current Report (Images, Data & Graphs)",
        data=zip_data,
        file_name=f"Defect_Report_{current_run['category_raw']}_Latest.zip",
        mime="application/zip",
        key="download_current"
    )

# --- TAB 2: HISTORY PANEL ---
with tab2:
    st.markdown("### Previous Inspections")
    if not history_runs:
        st.info("ℹ️ No previous runs found. Run your pipeline again to start building history.")
    else:
        # We use enumerate(history_runs) here to give each download button a unique key!
        for i, run in enumerate(history_runs):
            run_time = time.strftime('%b %d, %Y - %I:%M %p', time.localtime(run['mtime'])) # Formatted to business friendly time
            
            with st.expander(f"🕰️ {run['category']} | {run['defect']} | Processed: {run_time}"):

                h_cols = st.columns(4)
                h_paths = run["paths"]
                if os.path.exists(h_paths["orig"]): h_cols[0].image(Image.open(h_paths["orig"]), caption="Original Image", use_container_width=True)
                if os.path.exists(h_paths["box"]): h_cols[1].image(Image.open(h_paths["box"]), caption="Autoencoder", use_container_width=True)
                if os.path.exists(h_paths["padim"]): h_cols[2].image(Image.open(h_paths["padim"]), caption="PaDiM Overlay", use_container_width=True)
                if os.path.exists(h_paths["patch"]): h_cols[3].image(Image.open(h_paths["patch"]), caption="PatchCore Overlay", use_container_width=True)

                st.divider()

                # 🔥 ADDED HEATMAPS IN HISTORY
                st.markdown("#### 🌡️ Heatmaps")
                h2_cols = st.columns(3)

                if os.path.exists(h_paths["ae_heat"]):
                    h2_cols[0].image(Image.open(h_paths["ae_heat"]), caption="AE Heatmap", use_container_width=True)

                if os.path.exists(h_paths["padim_heat"]):
                    h2_cols[1].image(Image.open(h_paths["padim_heat"]), caption="PaDiM Heatmap", use_container_width=True)

                if os.path.exists(h_paths["patch_heat"]):
                    h2_cols[2].image(Image.open(h_paths["patch_heat"]), caption="PatchCore Heatmap", use_container_width=True)

                st.divider()

                st.markdown(f"#### 📊 Analytics")
                hist_metrics_file = os.path.join(RESULTS_PATH, f"{run['category_raw']}_evaluation.csv")

                if os.path.exists(hist_metrics_file):
                    st.dataframe(pd.read_csv(hist_metrics_file), use_container_width=True, hide_index=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True) # Spacer
                    col_h1, col_h2 = st.columns([1, 2])

                    hist_auroc_img = os.path.join(RESULTS_PATH, f"{run['category_raw']}_auroc_graph.png")
                    if os.path.exists(hist_auroc_img):
                        col_h1.image(Image.open(hist_auroc_img), caption="AUROC Performance", use_container_width=True)

                    hist_dist_cols = col_h2.columns(3)
                    for col, model in zip(hist_dist_cols, ["autoencoder", "padim", "patchcore"]):
                        hist_dist_img = os.path.join(RESULTS_PATH, f"{run['category_raw']}_{model}_score_distribution.png")
                        if os.path.exists(hist_dist_img):
                            col.image(Image.open(hist_dist_img), caption=f"{model.capitalize()} Distribution", use_container_width=True)
                else:
                    st.info("ℹ️ Analytics not found for this historical run.")
                
                st.markdown("<br>", unsafe_allow_html=True) # Spacer

                # 🌟 HISTORY RUN DOWNLOAD BUTTON
                hist_zip_data = create_run_zip(run)
                st.download_button(
                    label=f"🫧 Download Archive Report ({run_time})",
                    data=hist_zip_data,
                    file_name=f"Archive_Report_{run['category_raw']}_{int(run['mtime'])}.zip",
                    mime="application/zip",
                    key=f"download_hist_{i}" # MUST have a unique key inside a loop!
                )
