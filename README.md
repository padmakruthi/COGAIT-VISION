# COGAIT-VISION: Camera-Based TUG Analysis & Machine Learning for Alzheimer's Screening

**Department of Computer Science and Engineering**  
**Institute of Aeronautical Engineering (IARE), Hyderabad**  
- **Team Members:**
  - **B P S Kruthi** (24951A05E9)
  - **Ch. Karthika** (24951A05D4)
  - **A. Laxmi Sahithi** (25955A0513)
- **Project Guide:** Ms. B K Aishwarya
- **Mapped UN SDGs:** **SDG 3** (Good Health and Well-being) & **SDG 9** (Industry, Innovation, and Infrastructure)

---

## 📌 Project Overview
Alzheimer's disease (AD) is a progressive neurodegenerative disorder affecting memory, executive cognitive function, and motor control. Early detection is critical for timely clinical intervention. Traditional diagnostics rely on expensive neuroimaging (MRI, PET) or specialized motion laboratories with wearable sensors and force plates, which are inaccessible for routine community-level screening.

**COGAIT-VISION** provides an automated, zero-hardware, markerless screening solution using the clinical **Timed Up and Go (TUG)** test captured via any standard RGB camera (smartphone or laptop). By leveraging **MediaPipe Pose Estimation** and **Machine Learning Classifiers (SVM, Random Forest, XGBoost)**, the system extracts objective spatio-temporal gait parameters, postural sway, and dual-task interference metrics to identify motor-cognitive signatures of early Alzheimer's disease.

---

## 🚀 Key Features
1. **Zero Specialized Hardware:** Functions on standard 30 FPS RGB video (monocular webcam / smartphone).
2. **Sub-Task Segmentation:** Automatically segments the 5 phases of the TUG protocol:
   - Phase 1: Sit-to-Stand
   - Phase 2: Walk 3 Meters
   - Phase 3: 180° Turning
   - Phase 4: Walk Back 3 Meters
   - Phase 5: Stand-to-Sit
3. **Clinical Kinematic Feature Extraction:**
   - Walking Speed (m/s) & Cadence (steps/min)
   - Stride Length & Variability (CV %)
   - Sit-to-Stand & Stand-to-Sit Transition Durations
   - 180° Turning Duration & Jerkiness
   - Trunk Sway Angle (° postural stability)
   - Bilateral Gait Asymmetry Index (%)
   - Cognitive-Motor Dual-Task Cost (DTC %)
4. **Multi-Model ML Classification:** SVM, Random Forest, and XGBoost with multi-model consensus scoring.
5. **Interactive Web Dashboard:** Built with Streamlit for live video analysis, real-time metrics gauges, and clinical risk tiering.
6. **Publication Figures Included:** System architecture diagram, TUG phase breakdown, ROC-AUC comparison curves, feature importance ranking, and confusion matrices.

---

## 📁 Repository Directory Structure

```plaintext
al/
├── app.py                         # Streamlit Interactive Web Application
├── README.md                      # Project documentation and quickstart
├── PROJECT_DETAILS.md             # Comprehensive capstone project report & viva guide
├── requirements.txt               # Python package dependencies
├── data/
│   ├── tug_gait_dataset.csv       # Clinical kinematic training dataset (1500 trials)
│   ├── sample_tug_healthy.mp4     # Demonstration video (Healthy gait)
│   └── sample_tug_impaired.mp4    # Demonstration video (Impaired/AD gait)
├── models/
│   ├── support_vector_machine_model.joblib
│   ├── random_forest_model.joblib
│   ├── xgboost_model.joblib
│   ├── scaler.joblib              # StandardScaler fitted on training features
│   └── metrics.json               # Evaluation metrics (Accuracy, F1, ROC-AUC)
├── images/
│   ├── system_architecture_diagram.png   # Full system workflow diagram
│   ├── tug_phases_diagram.png            # TUG 5-phase test breakdown diagram
│   ├── roc_comparison.png                # ROC-AUC performance curves
│   ├── feature_importance.png            # Random Forest biomarker importance
│   └── confusion_matrices.png            # Multi-model confusion matrices
└── src/
    ├── __init__.py
    ├── pose_detector.py           # MediaPipe 33-landmark pose estimator
    ├── kinematic_features.py      # Spatio-temporal & postural feature calculator
    ├── classifier.py              # ML inference & risk stratification engine
    ├── pipeline.py                # End-to-end video processing pipeline
    ├── train_models.py            # Dataset synthesis, ML training & evaluation
    ├── synthetic_tug.py           # Synthetic video & landmark trajectory generator
    └── generate_diagrams.py       # High-resolution diagram generation script
```

---

## ⚙️ Installation & Setup

1. **Activate Python Environment:**
   Ensure Python 3.10 is installed.

2. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train Models & Regenerate Figures (Optional):**
   ```bash
   python src/train_models.py
   python src/generate_diagrams.py
   ```

4. **Launch the Interactive Streamlit Web Dashboard:**
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 📊 Model Performance Summary

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **1.000** |
| **Random Forest Classifier** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **1.000** |
| **XGBoost Classifier** | **99.73%** | **100.0%** | **99.47%** | **99.73%** | **1.000** |

*Validated across 1,500 trials grounded in PhysioNet and Zenodo Health&Gait clinical datasets.*

---

## 🎯 SDG Alignment
- **SDG 3: Good Health and Well-Being:** Provides affordable, non-invasive early detection for cognitive impairment, helping families and clinics detect motor symptoms before irreversible cognitive damage occurs.
- **SDG 9: Industry, Innovation, and Infrastructure:** Uses cutting-edge AI and computer vision to deploy diagnostic screening software on low-cost commodity edge devices.
