"""
Process User Uploaded TUG Videos: Normal Walk vs Abnormal Walk
Extracts MediaPipe landmarks, computes kinematics, runs ML models, and saves annotated videos.
"""

import os
import sys
import json
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline import TUGAnalysisPipeline

def process():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/videos", exist_ok=True)

    normal_path = "data/videos/normal walk.mp4"
    abnormal_path = "data/videos/abnormal.mp4"

    if not os.path.exists(normal_path) or not os.path.exists(abnormal_path):
        print("Videos not found in data/videos/")
        return

    pipeline = TUGAnalysisPipeline()

    print("\n" + "="*60)
    print("1. PROCESSING NORMAL WALK VIDEO...")
    print("="*60)
    res_normal = pipeline.process_video(
        normal_path,
        output_annotated_path="data/videos/annotated_normal.mp4",
        dual_task=False
    )
    print(f"Verdict: {res_normal['consensus_label']}")
    print(f"Risk Score: {res_normal['average_impairment_probability']}%")
    print(f"Risk Tier: {res_normal['risk_stratification']}")
    print("Model Breakdown:")
    for k, v in res_normal['model_breakdown'].items():
        print(f"  - {k}: {v['prediction']} ({v['probability']}%)")

    print("\n" + "="*60)
    print("2. PROCESSING ABNORMAL / ALZHEIMER SYMPTOMS VIDEO...")
    print("="*60)
    res_abnormal = pipeline.process_video(
        abnormal_path,
        output_annotated_path="data/videos/annotated_abnormal.mp4",
        dual_task=True
    )
    print(f"Verdict: {res_abnormal['consensus_label']}")
    print(f"Risk Score: {res_abnormal['average_impairment_probability']}%")
    print(f"Risk Tier: {res_abnormal['risk_stratification']}")
    print("Model Breakdown:")
    for k, v in res_abnormal['model_breakdown'].items():
        print(f"  - {k}: {v['prediction']} ({v['probability']}%)")

    # Generate Comparison Table
    df_compare = pd.DataFrame([
        {"Video": "Normal Walk (Healthy)", **res_normal['input_features'], "Verdict": res_normal['consensus_label'], "Risk Score (%)": res_normal['average_impairment_probability']},
        {"Video": "Abnormal (Alzheimer Symptoms)", **res_abnormal['input_features'], "Verdict": res_abnormal['consensus_label'], "Risk Score (%)": res_abnormal['average_impairment_probability']}
    ])
    df_compare.to_csv("reports/video_comparison_results.csv", index=False)
    print("\n" + "="*60)
    print("COMPARISON TABLE:")
    print("="*60)
    print(df_compare.T)

    # Save radar/bar comparison figure
    metrics_to_plot = [
        ("total_tug_duration", "TUG Duration (s)"),
        ("walking_speed", "Speed (m/s)"),
        ("cadence", "Cadence (steps/m)"),
        ("turning_duration", "Turn Time (s)"),
        ("trunk_sway_angle", "Trunk Sway (°)"),
        ("dual_task_cost", "Dual-Task Cost (%)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
    axes = axes.flatten()

    fn = res_normal['input_features']
    fa = res_abnormal['input_features']

    for i, (key, label) in enumerate(metrics_to_plot):
        ax = axes[i]
        vals = [fn.get(key, 0), fa.get(key, 0)]
        bars = ax.bar(["Normal Walk", "Abnormal Walk"], vals, color=["#27ae60", "#c0392b"], width=0.55)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Clinical Kinematic Comparison: User's Reference Videos", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    chart_path = "images/user_videos_comparison.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"\nGenerated comparison chart saved to: {chart_path}")

    # Save detailed JSON summary
    summary = {
        "normal_walk": res_normal,
        "abnormal_walk": res_abnormal
    }
    with open("reports/user_videos_evaluation.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("Saved evaluation report to reports/user_videos_evaluation.json")

if __name__ == "__main__":
    process()
