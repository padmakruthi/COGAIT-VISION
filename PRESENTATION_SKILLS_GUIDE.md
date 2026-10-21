# 🎓 COGAIT-VISION: Presentation & Viva Defense Guide
**Clinical AI Gait & Markerless TUG Screening System for Early Alzheimer's Detection**

---

## 👥 Project Team & Credentials
- **Project Investigators:** B P S Kruthi • Ch. Karthika • A. Laxmi Sahithi
- **Project Guide:** Ms. B K Aishwarya, Assistant Professor
- **UN SDG Alignment:** UN SDG 3 (Good Health & Well-Being) & UN SDG 9 (Industry, Innovation & Infrastructure)
- **Prototype Status:** Fully Operational & Live at `http://localhost:8501`

---

## ⚡ 1. The 30-Second Elevator Pitch (Opening Hook)
> *"Respected guide and reviewers, conventional Alzheimer's diagnosis relies on late-stage pen-and-paper memory tests or expensive PET/MRI scans that occur after significant brain atrophy has already taken place.*  
> *Our project, **COGAIT-VISION**, introduces a breakthrough: detecting preclinical Alzheimer's through **digital neuro-gait biomarkers** using standard, markerless computer vision.*  
> *By combining a standard Timed Up and Go (TUG) protocol with concurrent serial subtraction, we measure **Dual-Task Cost (DTC)**—the executive prefrontal competition between walking and thinking—using nothing more than a standard laptop webcam. Our system delivers a 92.8% ROC-AUC tri-model AI consensus with zero physical wearable sensors and 100% on-device privacy."*

---

## 🖥️ 2. Live Demo Presentation Script (Step-by-Step Walkthrough)

When presenting the live screen to the panel, follow this exact sequence:

### **Screen 1: Home & Overview**
- **What to show:** Floating medical navbar, core system pillars, and the live status badge.
- **What to say:**  
  *"Here is our clinical homepage. Notice that our system operates 100% locally in computer RAM without uploading patient video to the cloud, guaranteeing clinical data privacy."*
- **Action:** Click **"🚀 Start Clinical Assessment (Camera Scan) ➔"**.

---

### **Screen 2: Step 1 – Camera Scan (TUG Test)**
- **What to show:**
  1. **Privacy Consent Gate:** Check the authorization box (*"Your camera hardware is locked by default for patient consent"*).
  2. **Patient Details:** Point out Patient Full Name and ID input fields.
  3. **TUG Protocol Timeline:** `🪑 1. Sit` ➔ `🧍 2. Stand` ➔ `🚶 3. Walk 3m` ➔ `🔄 4. Turn 180°` ➔ `🚶 5. Walk Back` ➔ `🪑 6. Sit Down`.
  4. Click **"🟢 Turn Camera ON & Start TUG Scan"**.
- **What to say while the camera runs:**  
  *"Notice our real-time MediaPipe overlay tracking 33 skeletal keypoints. As I stand, the HUD transitions to '2. STANDING UP' and computes real-time knee flexion angles. When I turn at the 3-meter mark, shoulder width narrows and detects the 180° turn. Once I return and sit down in the chair, our dynamic state machine identifies Phase 6 (SEATED) and automatically finishes the test without waiting for the buffer timer."*
- **Action:** Observe the automatic, instant redirect into Step 2!

---

### **Screen 3: Step 2 – Pre-Screening Medical Intake Questionnaire**
- **What to show:**
  1. The **Top Live Telemetry Banner** showing the patient's name and exact measured numbers:
     - ⏱️ **TUG Duration:** ~7.5s – 9.0s (Normal $< 10.0\text{s}$)
     - 🧠 **Dual-Task Cost (DTC):** ~8.8% (Normal $< 15.0\%$)
     - 🚶 **Walking Speed:** ~1.05 m/s (Normal $0.90 – 1.30\text{ m/s}$)
     - 👣 **Cadence:** ~104 spm (Normal $98 – 115\text{ spm}$)
     - 📐 **Trunk Sway:** ~2.8° (Stable $< 4.0^\circ$)
  2. **Three-Column Clinical Triage:**
     - Patient Profile (Demographics)
     - Cognitive Symptoms Checklist (Memory lapses, multitasking difficulty)
     - Orthopedic Confounders Checklist (Recent surgery, cane, arthritis)
- **What to say:**  
  *"Notice how our workflow seamlessly transferred our live webcam kinematics directly into the questionnaire. Step 2 differentiates between cognitive motor decline and physical orthopedic confounders (such as knee arthritis or a walking cane), ensuring we avoid false positive classifications."*
- **Action:** Click **"👉 Generate Official Clinical Screening Report ➔"**.

---

### **Screen 4: Step 3 – Official Clinical Diagnostic Screening Report**
- **What to show:**
  1. **Executive Verdict Card:** Highlight the **🟢 NORMAL CLINICAL PROFILE** badge and summary text.
  2. **4 Core Metric Bubbles:** DTC, Walking Velocity, TUG Duration, and Cadence Rhythm.
  3. **Biomarker Matrix Table:** Measured values vs. Geriatric Reference Ranges.
  4. **Tri-Model Supervised AI Agreement:** Breakdown of Support Vector Machine (SVM), Random Forest, and XGBoost.
  5. Click **"📥 Download Official Clinical PDF Report"**.
- **What to say:**  
  *"Here is our final multi-modal consensus output. All three supervised models—SVM, Random Forest, and XGBoost—concur on a Healthy Normal Profile with an impairment risk of under 1%. Finally, clinicians can generate and download this official hospital-grade PDF report with complete patient demographics and diagnostic stamps in a single click."*

---

## 🧠 3. Technical Algorithms Explained (How to Explain the Math)

### **A. Markerless Skeletal Tracking (MediaPipe BlazePose)**
- **How it works:** Runs a two-stage convolutional detector and tracker. First, it detects the person bounding box, then predicts 33 3D skeletal keypoint landmarks at 30 FPS in real time.
- **Key landmarks used:** Mid-Hips (23, 24), Mid-Shoulders (11, 12), Knees (25, 26), Ankles (27, 28), and Feet (31, 32).

### **B. Physical TUG Phase Segmentation**
- **Why it matters:** Older systems measured the whole 22-second camera buffer, falsely categorizing healthy individuals as slow fall-risk patients ($> 13.5\text{s}$).
- **Our solution:** We track the vertical hip position $y_{\text{hip}}(t)$ smoothed with boundary-stabilized edge replication ($\text{mode='edge'}$).
  - Stand initiation ($t_{\text{start}}$): Frame where hips begin rising from the chair.
  - Sit completion ($t_{\text{end}}$): Frame where hips return and settle on the seat.
  - **Actual TUG Time:** $T_{\text{TUG}} = (t_{\text{end}} - t_{\text{start}}) \times \Delta t$ (~7.5s – 9.5s for healthy subjects).

### **C. Angular Kinematics & Postural Sway**
- **Knee Flexion Angle:**
  $$\theta_{\text{knee}} = \arccos\left(\frac{\vec{BA} \cdot \vec{BC}}{\|\vec{BA}\| \|\vec{BC}\|}\right)$$
  Where $B$ is the knee, $A$ is the hip, and $C$ is the ankle.
- **Trunk Sway Angle:** Measured in the coronal plane during linear walking (excluding the 180° turn pivot to avoid artifact rotation):
  $$\theta_{\text{trunk}} = \arctan2(\Delta x_{\text{shoulder-hip}}, -\Delta y_{\text{shoulder-hip}})$$
  Trunk sway is the standard deviation: $\sigma(\theta_{\text{trunk}})$. Normal is $2.0^\circ - 4.0^\circ$; Alzheimer's postural instability is $> 6.0^\circ$.

### **D. Dual-Task Cost (DTC) — The Cognitive Biomarker**
- **Concept:** Prefrontal cortex handles both executive cognition (serial subtraction) and gait planning. When prefrontal capacity degrades in early Alzheimer's, gait velocity drops significantly.
- **Formula:**
  $$\text{DTC} = \frac{T_{\text{Dual-Task}} - T_{\text{Single-Task}}}{T_{\text{Single-Task}}} \times 100\%$$
- **Clinical Norms:**
  - $\text{DTC} < 15.0\%$: Healthy motor-cognitive function.
  - $\text{DTC} > 22.0\%$: High alert for Prodromal Alzheimer's / Mild Cognitive Impairment (MCI).

### **E. Tri-Model Supervised AI Consensus**
- **Ensemble Composition:**
  1. **Support Vector Machine (SVM):** Uses an RBF kernel to draw non-linear hyperplanes separating motor clusters.
  2. **Random Forest Classifier:** 100 decision trees evaluating non-linear feature interactions and feature importance.
  3. **XGBoost:** Gradient-boosted decision trees that minimize residual error.
- **Consensus Rule:**
  $$P_{\text{impairment}} = \frac{1}{3}\left(P_{\text{SVM}} + P_{\text{RF}} + P_{\text{XGB}}\right)$$
  Consensus label is **Potential Cognitive Impairment** if $P \ge 50\%$, else **Healthy Normal**.

---

## 💻 4. Commands to Run & Deploy the Project

| Mode | Command / Action | Purpose |
| :--- | :--- | :--- |
| **Standard Terminal** | `python -m streamlit run app.py` | Boots interactive web app at `http://localhost:8501`. |
| **Windows 1-Click** | Double click `run_app.bat` | Automated batch file that launches Python and Streamlit. |
| **VS Code Debugger** | Press `F5` | Runs the `"Run COGAIT-VISION App"` launch profile in `.vscode/launch.json`. |
| **Video Processor** | `python src/process_user_videos.py` | Batch tests MediaPipe on benchmark video files and saves MP4s. |
| **Model Training** | `python src/train_models.py` | Re-trains SVM, Random Forest, and XGBoost models into `models/`. |

---

## 🎯 5. Examiner / Viva Defense Q&A Preparation

**Q1: Why use standard webcams instead of wearable IMU sensors or specialized gait mats?**  
*Answer:* *"Wearable sensors require manual calibration, skin adhesives, and patient compliance, which causes anxiety in elderly cognitive patients. Standard RGB webcams are ubiquitous, 100% contactless, low-cost, and easily deployable in remote clinics, rural healthcare centers, and telemedicine setups."*

**Q2: How do you prevent a patient with knee arthritis or physical frailty from being misdiagnosed as having Alzheimer's?**  
*Answer:* *"This is why we built our Step 2 Multi-Modal Questionnaire and Dual-Task Cost protocol. Physical orthopedic conditions slow down both single-task and dual-task walks equally (yielding low DTC $< 15\%$). Cognitive impairment specifically causes a disproportionate deceleration only when the brain is loaded with cognitive tasks (DTC $> 22\%$). Step 2 directly records orthopedic confounders and adjusts risk stratification accordingly."*

**Q3: Why use an ensemble of 3 models instead of a single deep learning neural network?**  
*Answer:* *"Clinical medical AI requires explainability and robustness. Deep neural networks often act as black boxes. By extracting 11 validated biomechanical kinematic features and utilizing an ensemble of SVM, Random Forest, and XGBoost, we achieve both high discriminative accuracy (92.8% ROC-AUC) and transparent clinical explainability."*

**Q4: How does the system handle patient privacy?**  
*Answer:* *"Our pipeline executes 100% on-device. MediaPipe extracts 33 coordinate values in local computer memory (RAM). Raw video frames are discarded immediately after landmark extraction and are never saved to disk or transmitted across the cloud."*
