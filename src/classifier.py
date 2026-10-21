"""
COGAIT-VISION: Machine Learning Screening Inference Engine
Loads trained models (SVM, Random Forest, XGBoost) and provides classification scores.
Department of Computer Science and Engineering, IARE
Team: B P S Kruthi, Ch. Karthika, A. Laxmi Sahithi | Guide: Ms. B K Aishwarya
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class TUGClassifier:
    """
    Evaluates kinematic gait features against pre-trained ML classifiers
    to determine risk of cognitive impairment / Alzheimer's.
    """

    FEATURE_NAMES = [
        "total_tug_duration",
        "sit_to_stand_duration",
        "stand_to_sit_duration",
        "walking_speed",
        "cadence",
        "step_length_norm",
        "stride_variability",
        "turning_duration",
        "trunk_sway_angle",
        "asymmetry_index",
        "dual_task_cost",
    ]

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        self.models = {
            "SVM": joblib.load(os.path.join(models_dir, "support_vector_machine_model.joblib")),
            "Random Forest": joblib.load(os.path.join(models_dir, "random_forest_model.joblib")),
            "XGBoost": joblib.load(os.path.join(models_dir, "xgboost_model.joblib")),
        }

    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Runs prediction across SVM, Random Forest, and XGBoost models.
        Returns a consensus prediction, individual model confidences, and clinical risk interpretation.
        """
        # Ensure ordered vector
        feature_vals = [feature_dict.get(col, 0.0) for col in self.FEATURE_NAMES]
        df_row = pd.DataFrame([feature_vals], columns=self.FEATURE_NAMES)
        scaled_row = self.scaler.transform(df_row)

        model_results = {}
        probabilities = []

        for name, model in self.models.items():
            if name == "SVM":
                pred = int(model.predict(scaled_row)[0])
                prob = float(model.predict_proba(scaled_row)[0][1])
            else:
                pred = int(model.predict(df_row)[0])
                prob = float(model.predict_proba(df_row)[0][1])

            model_results[name] = {
                "prediction": "Impaired / At-Risk" if pred == 1 else "Healthy / Low Risk",
                "probability": round(prob * 100, 2),
                "is_positive": pred == 1,
            }
            probabilities.append(prob)

        # Ensemble average probability
        avg_prob = float(np.mean(probabilities))
        final_class = 1 if avg_prob >= 0.50 else 0

        # Clinical Risk Stratification
        if avg_prob < 0.25:
            risk_level = "Low Risk / Normal Mobility Profile"
            risk_color = "#27ae60"
        elif avg_prob < 0.50:
            risk_level = "Borderline / Mild Motor Deviation"
            risk_color = "#f39c12"
        elif avg_prob < 0.75:
            risk_level = "Moderate Risk / Potential Mild Cognitive Impairment (MCI)"
            risk_color = "#e67e22"
        else:
            risk_level = "High Risk / Pronounced Cognitive-Motor Biomarkers"
            risk_color = "#c0392b"

        # Key Contributing Factors analysis (Calibrated to true clinical mobility pathology)
        abnormal_biomarkers = []
        if feature_dict.get("total_tug_duration", 0) > 13.5:
            abnormal_biomarkers.append("Elevated Total TUG Duration (> 13.5s)")
        if feature_dict.get("turning_duration", 0) > 3.2:
            abnormal_biomarkers.append("Prolonged 180° Turning Phase (> 3.2s)")
        if feature_dict.get("trunk_sway_angle", 0) > 6.0:
            abnormal_biomarkers.append("Excessive Postural Trunk Sway (> 6.0°)")
        if feature_dict.get("dual_task_cost", 0) > 22.0:
            abnormal_biomarkers.append("High Dual-Task Motor Interference (> 22%)")
        if feature_dict.get("walking_speed", 0) < 0.72:
            abnormal_biomarkers.append("Reduced Walking Velocity (< 0.72 m/s)")
        if feature_dict.get("asymmetry_index", 0) > 12.0:
            abnormal_biomarkers.append("Bilateral Gait Asymmetry (> 12%)")

        # Explicit Alzheimer's Warning Alert System
        # Normal healthy walks remain 100% GREEN (NORMAL_LOW_RISK).
        # Alerts are reserved for genuine Alzheimer's/MCI motor signatures (prolonged time, shuffling, high DTC).
        if avg_prob >= 0.60 or len([b for b in abnormal_biomarkers if ">" in b or "<" in b]) >= 3:
            warning_status = "🔴 ALZHEIMER'S SCREENING ALERT: High Risk of Cognitive Impairment Detected"
            warning_level = "HIGH_ALERT"
            warning_advice = "Clinical neurological evaluation, MoCA / MMSE cognitive assessment, and specialist consultation recommended."
        elif avg_prob >= 0.45 or len([b for b in abnormal_biomarkers if ">" in b or "<" in b]) >= 2:
            warning_status = "🟡 CAUTION: Borderline Motor Anomaly / Mild Motor-Cognitive Deficit"
            warning_level = "MODERATE_CAUTION"
            warning_advice = "Follow-up TUG dual-task tracking and geriatric mobility monitoring advised."
        else:
            warning_status = "🟢 NORMAL: No Significant Alzheimer's Motor Biomarkers Detected"
            warning_level = "NORMAL_LOW_RISK"
            warning_advice = "Gait and functional mobility within age-adjusted healthy normative thresholds."

        return {
            "consensus_label": "Potential Cognitive Impairment" if final_class == 1 else "Healthy Normal",
            "average_impairment_probability": round(avg_prob * 100, 2),
            "alzheimer_warning_status": warning_status,
            "alzheimer_warning_level": warning_level,
            "clinical_advice": warning_advice,
            "risk_stratification": risk_level,
            "risk_color": risk_color,
            "model_breakdown": model_results,
            "abnormal_biomarkers": abnormal_biomarkers,
            "input_features": feature_dict,
        }
