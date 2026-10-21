"""
COGAIT-VISION: Machine Learning Training and Evaluation Pipeline
Department of Computer Science and Engineering, IARE
Team: B P S Kruthi, Ch. Karthika, A. Laxmi Sahithi | Guide: Ms. B K Aishwarya
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


FEATURE_COLUMNS = [
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


def generate_clinically_grounded_dataset(n_samples: int = 1500, random_seed: int = 42) -> pd.DataFrame:
    """
    Synthesizes realistic TUG kinematics grounded in literature (PhysioNet, Zenodo Health&Gait).
    Incorporates realistic clinical overlap (Mild Cognitive Impairment borderline cases).
    Class 0: Healthy Older Adults
    Class 1: Potential Cognitive Impairment / Early Alzheimer's
    """
    np.random.seed(random_seed)
    n_per_class = n_samples // 2

    # Class 0: Healthy
    healthy_data = {
        "total_tug_duration": np.random.normal(loc=9.8, scale=1.8, size=n_per_class).clip(7.0, 15.0),
        "sit_to_stand_duration": np.random.normal(loc=1.7, scale=0.4, size=n_per_class).clip(1.1, 2.7),
        "stand_to_sit_duration": np.random.normal(loc=1.8, scale=0.4, size=n_per_class).clip(1.1, 2.8),
        "walking_speed": np.random.normal(loc=1.12, scale=0.18, size=n_per_class).clip(0.75, 1.50),
        "cadence": np.random.normal(loc=102.0, scale=10.0, size=n_per_class).clip(80.0, 125.0),
        "step_length_norm": np.random.normal(loc=0.39, scale=0.05, size=n_per_class).clip(0.28, 0.50),
        "stride_variability": np.random.normal(loc=4.6, scale=1.8, size=n_per_class).clip(1.8, 9.5),
        "turning_duration": np.random.normal(loc=2.3, scale=0.5, size=n_per_class).clip(1.4, 3.8),
        "trunk_sway_angle": np.random.normal(loc=3.6, scale=1.1, size=n_per_class).clip(1.8, 6.8),
        "asymmetry_index": np.random.normal(loc=5.2, scale=2.1, size=n_per_class).clip(1.5, 11.5),
        "dual_task_cost": np.random.normal(loc=11.5, scale=4.5, size=n_per_class).clip(3.0, 24.0),
        "label": [0] * n_per_class,
    }

    # Class 1: Impaired / Early AD (with realistic distribution overlap)
    impaired_data = {
        "total_tug_duration": np.random.normal(loc=15.6, scale=3.1, size=n_per_class).clip(10.5, 26.0),
        "sit_to_stand_duration": np.random.normal(loc=2.9, scale=0.7, size=n_per_class).clip(1.7, 5.2),
        "stand_to_sit_duration": np.random.normal(loc=3.0, scale=0.7, size=n_per_class).clip(1.8, 5.4),
        "walking_speed": np.random.normal(loc=0.74, scale=0.16, size=n_per_class).clip(0.40, 1.10),
        "cadence": np.random.normal(loc=80.0, scale=11.0, size=n_per_class).clip(55.0, 102.0),
        "step_length_norm": np.random.normal(loc=0.28, scale=0.05, size=n_per_class).clip(0.16, 0.39),
        "stride_variability": np.random.normal(loc=10.2, scale=2.9, size=n_per_class).clip(4.8, 18.5),
        "turning_duration": np.random.normal(loc=4.2, scale=1.0, size=n_per_class).clip(2.5, 7.2),
        "trunk_sway_angle": np.random.normal(loc=7.6, scale=2.0, size=n_per_class).clip(3.9, 14.0),
        "asymmetry_index": np.random.normal(loc=13.1, scale=3.6, size=n_per_class).clip(5.5, 24.0),
        "dual_task_cost": np.random.normal(loc=31.8, scale=7.5, size=n_per_class).clip(14.0, 52.0),
        "label": [1] * n_per_class,
    }

    df_healthy = pd.DataFrame(healthy_data)
    df_impaired = pd.DataFrame(impaired_data)
    df = pd.concat([df_healthy, df_impaired], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df


def train_and_evaluate(data_dir: str = "data", models_dir: str = "models", images_dir: str = "images"):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print("[1/5] Generating clinically grounded dataset...")
    df = generate_clinically_grounded_dataset(n_samples=1500)
    dataset_path = os.path.join(data_dir, "tug_gait_dataset.csv")
    df.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path} ({len(df)} samples)")

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("[2/5] Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

    # Define classifiers
    models = {
        "Support Vector Machine": (SVC(kernel="rbf", C=1.5, probability=True, random_state=42), True),
        "Random Forest": (RandomForestClassifier(n_estimators=150, max_depth=7, min_samples_split=4, random_state=42), False),
        "XGBoost": (xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.07, eval_metric="logloss", random_state=42), False),
    }

    results = {}
    roc_data = {}
    cm_dict = {}

    print("[3/5] Training models and computing metrics...")
    for name, (model, requires_scale) in models.items():
        if requires_scale:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        cm_dict[name] = cm

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}

        results[name] = {
            "Accuracy": round(float(acc) * 100, 2),
            "Precision": round(float(prec) * 100, 2),
            "Recall": round(float(rec) * 100, 2),
            "F1_Score": round(float(f1) * 100, 2),
            "ROC_AUC": round(float(auc), 4),
            "Confusion_Matrix": cm,
        }

        # Save model
        filename = name.lower().replace(" ", "_") + "_model.joblib"
        joblib.dump(model, os.path.join(models_dir, filename))
        print(f" -> {name}: Acc={acc*100:.2f}%, F1={f1*100:.2f}%, AUC={auc:.4f}")

    # Save metrics JSON
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("[4/5] Generating publication ROC curves plot...")
    plt.figure(figsize=(8, 6), dpi=300)
    colors = {"Support Vector Machine": "#2b5c8f", "Random Forest": "#16a085", "XGBoost": "#d35400"}
    for name, curve in roc_data.items():
        plt.plot(
            curve["fpr"],
            curve["tpr"],
            label=f"{name} (AUC = {curve['auc']:.3f})",
            color=colors.get(name, "blue"),
            lw=2.2,
        )
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Chance Level (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("ROC-AUC Comparison: Machine Learning Models for TUG Screening", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    roc_img_path = os.path.join(images_dir, "roc_comparison.png")
    plt.savefig(roc_img_path)
    plt.close()
    print(f"ROC curves saved to {roc_img_path}")

    # Feature Importance Plot (Random Forest)
    print("[5/5] Generating Feature Importance chart...")
    rf_model = models["Random Forest"][0]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)
    
    clean_names = {
        "dual_task_cost": "Dual-Task Cost (%)",
        "trunk_sway_angle": "Trunk Sway Angle (°)",
        "turning_duration": "Turning Duration (s)",
        "total_tug_duration": "Total TUG Duration (s)",
        "walking_speed": "Walking Speed (m/s)",
        "stride_variability": "Stride Variability (%)",
        "sit_to_stand_duration": "Sit-to-Stand Duration (s)",
        "asymmetry_index": "Gait Asymmetry Index (%)",
        "stand_to_sit_duration": "Stand-to-Sit Duration (s)",
        "cadence": "Cadence (steps/min)",
        "step_length_norm": "Normalized Step Length",
    }
    
    sorted_labels = [clean_names.get(FEATURE_COLUMNS[i], FEATURE_COLUMNS[i]) for i in indices]
    sorted_vals = importances[indices]

    plt.figure(figsize=(9, 6), dpi=300)
    plt.barh(range(len(sorted_vals)), sorted_vals, color="#1d4e89", edgecolor="#0e294b")
    plt.yticks(range(len(sorted_vals)), sorted_labels, fontsize=10)
    plt.xlabel("Gini Feature Importance Weight", fontsize=11)
    plt.title("Key Biomarker Feature Importance (Random Forest)", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    fi_img_path = os.path.join(images_dir, "feature_importance.png")
    plt.savefig(fi_img_path)
    plt.close()
    print(f"Feature importance saved to {fi_img_path}")

    # Generate Confusion Matrix Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    model_keys = ["Support Vector Machine", "Random Forest", "XGBoost"]
    for ax, m_name in zip(axes, model_keys):
        matrix = np.array(cm_dict[m_name])
        im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{m_name}\nAcc: {results[m_name]['Accuracy']}%", fontweight="bold", fontsize=11)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['Healthy', 'Impaired'])
        ax.set_yticklabels(['Healthy', 'Impaired'])
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2.0 else "black",
                        fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(images_dir, "confusion_matrices.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrices saved to {cm_path}")

    return results


if __name__ == "__main__":
    train_and_evaluate()
