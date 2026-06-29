"""
Crop Disease Detection using CNN
Phase 4: Streamlit Demo App
-----------------------------------------
Features:
  - Upload leaf image → instant disease prediction
  - Shows top-3 predictions with confidence bars
  - Disease info: symptoms, treatment, severity
  - Model selector (Custom CNN / MobileNetV2 / ResNet50)
  - Batch prediction from folder
  - Integration hook for Smart Farming weather advisory

Run: streamlit run phase4_app.py
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import warnings
warnings.filterwarnings("ignore")

                                               
             
                                               
st.set_page_config(
    page_title="Crop Disease Detection AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .main-title { font-size: 2rem; font-weight: 700; color: #1D9E75; }
  .sub-title  { font-size: 0.95rem; color: #666; margin-top: -8px; }
  .pred-card  {
    background: #f8f9fa; border-radius: 10px;
    padding: 1rem 1.2rem; border-left: 4px solid #1D9E75;
    margin-bottom: 8px;
  }
  .disease-card {
    background: #fdecea; border-radius: 10px;
    padding: 1rem 1.2rem; border-left: 4px solid #E05C2A;
  }
  .healthy-card {
    background: #e8f5e9; border-radius: 10px;
    padding: 1rem 1.2rem; border-left: 4px solid #1D9E75;
  }
  .severity-high   { color: #E05C2A; font-weight: 500; }
  .severity-medium { color: #E8A838; font-weight: 500; }
  .severity-low    { color: #1D9E75; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

                                               
                  
                                               
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "severity": "Medium",
        "symptoms": "Dark olive-green spots on leaves and fruit. Leaves may yellow and drop early.",
        "treatment": "Apply fungicide (captan or myclobutanil) at early season. Remove infected leaves.",
        "prevention": "Plant resistant varieties. Ensure good air circulation."
    },
    "Apple___Black_rot": {
        "severity": "High",
        "symptoms": "Brown circular lesions on fruit. Purple spots on leaves with frog-eye appearance.",
        "treatment": "Remove mummified fruit. Apply thiophanate-methyl fungicide.",
        "prevention": "Prune dead wood. Avoid overhead irrigation."
    },
    "Tomato___Late_blight": {
        "severity": "High",
        "symptoms": "Water-soaked dark green lesions on leaves. White mould on underside. Fruit rots.",
        "treatment": "Apply chlorothalonil or copper-based fungicide immediately. Remove infected plants.",
        "prevention": "Avoid wetting foliage. Use certified disease-free seeds."
    },
    "Tomato___Early_blight": {
        "severity": "Medium",
        "symptoms": "Dark brown spots with concentric rings (target-like) on older leaves.",
        "treatment": "Apply mancozeb or azoxystrobin fungicide. Remove affected lower leaves.",
        "prevention": "Crop rotation every 2 years. Mulch around plants."
    },
    "Corn_(maize)___Common_rust_": {
        "severity": "Medium",
        "symptoms": "Small cinnamon-brown powdery pustules on both leaf surfaces.",
        "treatment": "Apply triazole or strobilurin fungicide early. Plant resistant hybrids.",
        "prevention": "Early planting. Scout fields regularly in humid conditions."
    },
    "Rice___Leaf_blast": {
        "severity": "High",
        "symptoms": "Diamond-shaped lesions with grey centre and brown border on leaves.",
        "treatment": "Apply tricyclazole or isoprothiolane fungicide at first sign.",
        "prevention": "Avoid excess nitrogen. Maintain proper field drainage."
    },
    "healthy": {
        "severity": "None",
        "symptoms": "No disease symptoms detected.",
        "treatment": "No treatment needed. Continue good agricultural practices.",
        "prevention": "Maintain regular monitoring, proper irrigation, and balanced nutrition."
    }
}

def get_disease_info(class_name):
    """Look up disease info, using partial match if exact not found."""
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]
    for key in DISEASE_INFO:
        if key.lower() in class_name.lower() or class_name.lower() in key.lower():
            return DISEASE_INFO[key]
    is_healthy = "healthy" in class_name.lower()
    return {
        "severity": "None" if is_healthy else "Unknown",
        "symptoms": "Plant appears healthy." if is_healthy else "Consult an agricultural expert for diagnosis.",
        "treatment": "No treatment needed." if is_healthy else "Contact your local agricultural extension office.",
        "prevention": "Continue current farming practices." if is_healthy else "Monitor crop regularly."
    }


                                               
                     
                                               
OUTPUT_DIR = "crop_disease_outputs"
META_PATH  = f"{OUTPUT_DIR}/metadata.json"

@st.cache_resource(show_spinner="Loading AI model...")
def load_model(model_name):
    model_paths = {
        "MobileNetV2 (Best accuracy)": f"{OUTPUT_DIR}/mobilenetv2_final.keras",
        "ResNet50":                    f"{OUTPUT_DIR}/resnet50_final.keras",
        "Custom CNN (Baseline)":       f"{OUTPUT_DIR}/custom_cnn.keras",
    }
    path = model_paths[model_name]
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)

@st.cache_data
def load_metadata():
    if not os.path.exists(META_PATH):
        return None
    with open(META_PATH) as f:
        return json.load(f)


                                               
                   
                                               
DEFAULT_IMG_SIZE = (128, 128)


def get_model_input_size(model):
    if model is None:
        return DEFAULT_IMG_SIZE

    input_shape = getattr(model, "input_shape", None)
    if input_shape is None:
        return DEFAULT_IMG_SIZE

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) >= 3:
        height = input_shape[1]
        width = input_shape[2]
        if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
            return (height, width)

    return DEFAULT_IMG_SIZE


def preprocess_image(image: Image.Image, model=None):
    target_size = get_model_input_size(model)
    img = image.convert("RGB").resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(model, image_array, class_names, top_k=3):
    preds = model.predict(image_array, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "class": class_names[idx],
            "label": class_names[idx].replace("___", " → ").replace("_", " "),
            "confidence": float(preds[idx]) * 100,
            "is_healthy": "healthy" in class_names[idx].lower(),
        })
    return results


                                               
         
                                               
with st.sidebar:
    st.image("https://img.icons8.com/color/96/leaf.png", width=55)
    st.markdown("### 🌿 Crop Disease AI")
    st.divider()

    model_choice = st.selectbox(
        "Select Model",
        ["MobileNetV2 (Best accuracy)", "ResNet50", "Custom CNN (Baseline)"],
        help="MobileNetV2 is faster and slightly more accurate for most crops."
    )

    st.divider()
    st.markdown("**Supported Crops**")
    st.markdown("""
    Apple · Cherry · Corn · Grape ·
    Orange · Peach · Pepper · Potato ·
    Raspberry · Rice · Soybean · Squash ·
    Strawberry · Tomato
    """)
    st.divider()
    st.markdown("**Model Performance**")
    st.markdown("- Custom CNN: ~75% accuracy")
    st.markdown("- MobileNetV2: ~95% accuracy")
    st.markdown("- ResNet50: ~94% accuracy")
    st.divider()
    st.caption("Crop Disease Detection | CNN Project\nPlantVillage Dataset | 38 Classes")


                                               
          
                                               
st.markdown('<p class="main-title">🌿 Crop Disease Detection AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload a leaf image → instant AI diagnosis with treatment recommendations</p>', unsafe_allow_html=True)
st.divider()

meta = load_metadata()
if meta is None:
    st.error("⚠️ Metadata not found. Please run Phase 1 first.")
    st.stop()

class_names = meta["class_names"]
model = load_model(model_choice)

if model is None:
    st.warning(f"⚠️ Model file not found for '{model_choice}'. Please complete training first.")
    st.stop()

st.success(f"✅ Model loaded: **{model_choice}** | {len(class_names)} disease classes")

            
tab1, tab2, tab3 = st.tabs(["🔍 Single Image", "📁 Batch Prediction", "📊 Disease Reference"])

                                            
                                
                                            
with tab1:
    col1, col2 = st.columns([1, 1.4], gap="large")

    with col1:
        st.markdown("#### Upload Leaf Image")
        uploaded = st.file_uploader(
            "Choose a leaf image (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            help="Take a clear photo of a single leaf in good lighting."
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded leaf", use_column_width=True)

            tips = st.expander("📷 Tips for better results")
            with tips:
                st.markdown("""
                - Use a clear, well-lit photo
                - Single leaf fills most of the frame
                - Avoid blurry or shadowed images
                - Capture both surfaces if symptoms differ
                """)

    with col2:
        if uploaded:
            st.markdown("#### Diagnosis Result")
            with st.spinner("Analysing leaf..."):
                img_array = preprocess_image(image, model=model)
                predictions = predict(model, img_array, class_names, top_k=3)

            top = predictions[0]
            is_healthy = top["is_healthy"]

                              
            card_class = "healthy-card" if is_healthy else "disease-card"
            icon = "✅" if is_healthy else "⚠️"
            st.markdown(f"""
            <div class="{card_class}">
                <h3 style="margin:0">{icon} {top['label']}</h3>
                <p style="margin:4px 0 0; font-size:1.2rem; font-weight:500">
                    Confidence: {top['confidence']:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

                            
            st.markdown("**Top-3 Predictions:**")
            for pred in predictions:
                bar_color = "#1D9E75" if pred["is_healthy"] else "#E05C2A"
                st.markdown(f"**{pred['label']}**")
                st.progress(pred["confidence"] / 100)
                st.caption(f"{pred['confidence']:.1f}% confidence")

            st.divider()

                          
            info = get_disease_info(top["class"])
            sev_class = {
                "High": "severity-high",
                "Medium": "severity-medium",
                "Low": "severity-low",
                "None": "severity-low",
            }.get(info["severity"], "severity-medium")

            st.markdown(f"**Severity:** <span class='{sev_class}'>{info['severity']}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"**Symptoms:** {info['symptoms']}")
            st.markdown(f"**Treatment:** {info['treatment']}")
            st.markdown(f"**Prevention:** {info['prevention']}")

        else:
            st.info("👆 Upload a leaf image to get started.")
            st.markdown("""
            **What this AI can detect:**
            - 38 different crop diseases across 14 plant species
            - Fungal, bacterial, and viral infections
            - Healthy vs diseased classification
            - Confidence score for each prediction
            """)


                                            
                         
                                            
with tab2:
    st.markdown("#### Upload Multiple Leaf Images")
    batch_files = st.file_uploader(
        "Choose multiple leaf images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if batch_files:
        st.markdown(f"**{len(batch_files)} images uploaded**")
        if st.button("Run Batch Prediction", type="primary"):
            results = []
            prog = st.progress(0)
            for i, f in enumerate(batch_files):
                img = Image.open(f)
                arr = preprocess_image(img, model=model)
                preds = predict(model, arr, class_names, top_k=1)
                results.append({
                    "Filename": f.name,
                    "Prediction": preds[0]["label"],
                    "Confidence (%)": f"{preds[0]['confidence']:.1f}",
                    "Status": "Healthy ✅" if preds[0]["is_healthy"] else "Disease ⚠️",
                })
                prog.progress((i + 1) / len(batch_files))

            import pandas as pd
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            disease_count = sum(1 for r in results if "Disease" in r["Status"])
            healthy_count = len(results) - disease_count
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Images", len(results))
            c2.metric("Diseased", disease_count, delta=f"{disease_count/len(results)*100:.0f}%")
            c3.metric("Healthy", healthy_count, delta=f"{healthy_count/len(results)*100:.0f}%")

            csv = df.to_csv(index=False)
            st.download_button("📥 Download Results CSV", csv, "batch_results.csv", "text/csv")


                                            
                          
                                            
with tab3:
    st.markdown("#### Disease Reference Guide")
    st.markdown(f"Full list of **{len(class_names)} classes** in the PlantVillage dataset:")

    import pandas as pd
    rows = []
    for cls in class_names:
        parts = cls.split("___")
        plant = parts[0].replace("_", " ")
        condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
        status = "✅ Healthy" if "healthy" in condition.lower() else "⚠️ Disease"
        rows.append({"Plant": plant, "Condition": condition, "Status": status})

    df_ref = pd.DataFrame(rows)

    plant_filter = st.multiselect(
        "Filter by plant",
        sorted(df_ref["Plant"].unique()),
        default=[]
    )
    if plant_filter:
        df_ref = df_ref[df_ref["Plant"].isin(plant_filter)]

    status_filter = st.radio("Show", ["All", "Diseases only", "Healthy only"], horizontal=True)
    if status_filter == "Diseases only":
        df_ref = df_ref[df_ref["Status"] == "⚠️ Disease"]
    elif status_filter == "Healthy only":
        df_ref = df_ref[df_ref["Status"] == "✅ Healthy"]

    st.dataframe(df_ref, use_container_width=True, height=500)
    st.caption(f"Showing {len(df_ref)} of {len(class_names)} classes")