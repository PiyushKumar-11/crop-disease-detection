# 🌿 Crop Disease Detection using CNN

> Deep Learning Project | Computer Vision | Transfer Learning

A deep learning system that detects 38 different crop diseases from leaf images using Convolutional Neural Networks. Built with TensorFlow/Keras using the PlantVillage dataset.

---

## 📌 Project Overview

Farmers can upload a photo of a crop leaf and instantly receive:
- Disease identification with confidence score
- Top-3 possible diagnoses
- Symptoms description
- Chemical and organic treatment options
- Prevention tips

### Key Features
- 🔍 **38-class disease classification** across 14 crop types
- 🧠 **Two-stage transfer learning** (MobileNetV2 fine-tuned on PlantVillage)
- 📊 **Custom CNN baseline** for comparison
- 🌿 **Full disease database** with treatment and prevention info
- 🖥️ **Streamlit web app** for live demo

---

## 🌱 Supported Crops & Diseases

| Crop | Conditions Detected |
|---|---|
| Apple | Apple Scab, Black Rot, Cedar Rust, Healthy |
| Corn | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, TYLCV, Mosaic Virus, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Grape | Black Rot, Esca, Leaf Blight, Healthy |
| + 9 more | Cherry, Peach, Pepper, Orange, Strawberry, Squash, Soybean, Blueberry, Raspberry |

---

## 📊 Dataset

| Property | Details |
|---|---|
| Name | [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) |
| Source | Kaggle |
| Total images | ~54,000 |
| Classes | 38 (disease + healthy combinations) |
| Split | 70% train / 15% val / 15% test |
| Image size | 128×128 (resized for memory efficiency) |

---

## 🤖 Models

### Model Comparison

| Model | Architecture | Test Accuracy | Top-3 Accuracy |
|---|---|---|---|
| Custom CNN | 3-block CNN from scratch | ~75% | ~92% |
| **MobileNetV2** | **Transfer learning (ImageNet)** | **~95%** | **~99%** ⭐ |

### MobileNetV2 Training Strategy
- **Stage 1** (Head only): Train custom classification head, base frozen — 8 epochs
- **Stage 2** (Fine-tuning): Unfreeze top 30 layers, retrain at LR=1e-5 — 20 epochs

---

## 🗂️ Project Structure

```
crop-disease-detection/
│
├── crop_disease_phase1_setup.py           # Dataset analysis, splitting, pipelines
├── crop_disease_phase2_custom_cnn.py      # Custom CNN from scratch
├── crop_disease_phase3_colab.py           # MobileNetV2 transfer learning (Colab)
├── crop_disease_phase4_app.py             # Streamlit demo app
│
├── requirements_crop_disease.txt          # Python dependencies
├── .gitignore                             # Files excluded from repo
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/crop-disease-detection.git
cd crop-disease-detection
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements_crop_disease.txt
```

### 4. Download PlantVillage dataset
```bash
# Option A: Kaggle CLI
pip install kaggle
kaggle datasets download -d emmarex/plantdisease --unzip -p ./PlantVillage

# Option B: Manual download from
# https://www.kaggle.com/datasets/emmarex/plantdisease
```

---

## 🚀 Running the Project

### Local training (Phase 1 & 2)
```bash
# Phase 1 — Dataset setup & preprocessing
python crop_disease_phase1_setup.py

# Phase 2 — Train custom CNN baseline
python crop_disease_phase2_custom_cnn.py
```

### Colab training (Phase 3 — MobileNetV2)
```
1. Upload crop_disease_split/ folder to Google Drive
2. Open crop_disease_phase3_colab.py in Google Colab
3. Set runtime to T4 GPU
4. Run cells in order
5. Download mobilenetv2_final.keras from Drive
```

### Run the demo app (Phase 4)
```bash
# Place mobilenetv2_final.keras in crop_disease_outputs/
# Place class_names.json in crop_disease_outputs/

streamlit run crop_disease_phase4_app.py
```

---

## 🖥️ App Features

Upload any leaf image and get:

```
🔬 Diagnosis Result
  ⚠️ Tomato Late Blight          Confidence: 97.3%

Top-3 Predictions:
  🥇 Tomato → Late blight         97.3%
  🥈 Tomato → Early blight         2.1%
  🥉 Tomato → Healthy              0.6%

📋 Disease Info
  Cause     : Oomycete — Phytophthora infestans
  Severity  : HIGH
  Spreads   : Wind-driven spores
  Symptoms  : Water-soaked lesions...
  Treatment : Chlorothalonil, mancozeb...
  Organic   : Copper-based fungicide...
  Prevention: Avoid overhead irrigation...
```

---

## 💻 Training on Google Colab (Free T4 GPU)

Memory optimisations applied for free Colab T4:

| Setting | Original | Optimised |
|---|---|---|
| Image size | 224×224 | 128×128 |
| Batch size | 32 | 16 |
| Data caching | `.cache()` | Removed (streaming) |
| GPU memory | Default | Growth mode enabled |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Deep learning | `TensorFlow 2.15`, `Keras` |
| Base model | `MobileNetV2` (ImageNet pretrained) |
| Image processing | `Pillow`, `OpenCV` |
| Evaluation | `scikit-learn` |
| Visualisation | `matplotlib`, `seaborn` |
| Dashboard | `streamlit` |

---

## 📁 Output Files

```
crop_disease_outputs/
  ├── custom_cnn.keras                    ← Phase 2 model
  ├── mobilenetv2_final.keras             ← Phase 3 model (best)
  ├── class_names.json                    ← 38 class labels
  ├── results_summary.json                ← Accuracy comparison
  ├── training_history.png                ← Loss/accuracy curves
  ├── confusion_matrix.png                ← Per-class performance
  ├── model_comparison.png                ← CNN vs MobileNetV2
  └── sample_predictions.png             ← Visual prediction grid
```

---

## 👤 Author

**Piyush** — Deep Learning Project, 2025
Bihar, India

---

## 📄 License

This project is for educational purposes.
Dataset: PlantVillage (open access for research use).
