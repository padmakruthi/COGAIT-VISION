"""
COGAIT-VISION: Publication Diagrams & Visual Generation Module
Generates high-resolution architecture and TUG phase diagrams for review slides and documentation.
Department of Computer Science and Engineering, IARE
Team: B P S Kruthi, Ch. Karthika, A. Laxmi Sahithi | Guide: Ms. B K Aishwarya
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Arrow


def create_tug_phases_diagram(output_path: str = "images/tug_phases_diagram.png"):
    """Creates a visual breakdown of the Timed Up and Go (TUG) Protocol."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Title
    ax.text(7, 5.5, "Standardized Timed Up and Go (TUG) Test Sub-Tasks", 
            fontsize=16, fontweight="bold", ha="center", va="center", color="#1a252f")
    ax.text(7, 5.1, "Monocular Camera Video Capture (30 FPS) with 33 Skeletal Keypoint Tracking", 
            fontsize=11, ha="center", va="center", color="#555555", style="italic")

    phases = [
        {"num": "Phase 1", "title": "Sit-to-Stand", "desc": "Chair push-off\nTorso angle shift\nTransition: 1.2-2.0s", "color": "#3498db"},
        {"num": "Phase 2", "title": "Walk 3 Meters", "desc": "Steady acceleration\nCadence & stride\nSpeed: 1.0-1.4 m/s", "color": "#2ecc71"},
        {"num": "Phase 3", "title": "180° Turn", "desc": "Mid-point turn\nShoulder crossing\nDuration: 1.5-2.5s", "color": "#e67e22"},
        {"num": "Phase 4", "title": "Walk Back 3m", "desc": "Return trajectory\nStride variability\nDual-task evaluation", "color": "#9b59b6"},
        {"num": "Phase 5", "title": "Stand-to-Sit", "desc": "Controlled deceleration\nHip descent\nTransition: 1.3-2.1s", "color": "#e74c3c"}
    ]

    box_width = 2.2
    box_height = 3.2
    y_pos = 1.2
    
    for i, p in enumerate(phases):
        x_pos = 0.8 + i * 2.6
        
        # Outer card
        rect = FancyBboxPatch((x_pos, y_pos), box_width, box_height,
                              boxstyle="round,pad=0.1,rounding_size=0.2",
                              facecolor="#fdfefe", edgecolor=p["color"], linewidth=2.5,
                              alpha=0.95)
        ax.add_patch(rect)
        
        # Header banner
        header = FancyBboxPatch((x_pos, y_pos + box_height - 0.7), box_width, 0.7,
                                boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor=p["color"], edgecolor=p["color"])
        ax.add_patch(header)
        ax.text(x_pos + box_width/2, y_pos + box_height - 0.35, p["num"],
                fontsize=11, fontweight="bold", color="white", ha="center", va="center")
        
        # Title
        ax.text(x_pos + box_width/2, y_pos + box_height - 1.05, p["title"],
                fontsize=12, fontweight="bold", color="#2c3e50", ha="center", va="center")
        
        # Description
        ax.text(x_pos + box_width/2, y_pos + 1.0, p["desc"],
                fontsize=9.5, color="#34495e", ha="center", va="center", linespacing=1.4)
        
        # Arrow connecting phases
        if i < len(phases) - 1:
            ax.annotate("", xy=(x_pos + box_width + 0.35, y_pos + box_height/2),
                        xytext=(x_pos + box_width + 0.05, y_pos + box_height/2),
                        arrowprops=dict(facecolor="#7f8c8d", edgecolor="#7f8c8d", width=2, headwidth=7))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Created: {output_path}")


def create_system_architecture_diagram(output_path: str = "images/system_architecture_diagram.png"):
    """Creates a publication-style system architecture diagram."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Title
    ax.text(7.5, 7.0, "COGAIT-VISION: End-to-End System Architecture", 
            fontsize=17, fontweight="bold", ha="center", va="center", color="#0f2b48")
    ax.text(7.5, 6.6, "Camera-Based TUG Kinematic Analysis and Multi-Model Screening Pipeline", 
            fontsize=11, ha="center", va="center", color="#4a5568")

    stages = [
        {
            "step": "Stage 1", "title": "Video Acquisition",
            "items": ["• RGB Camera (Smartphone / Laptop)", "• Monocular Stream at 30 FPS", "• 3-Meter TUG Walking Path", "• Single / Dual-Task Protocol"],
            "color": "#1abc9c", "x": 0.8
        },
        {
            "step": "Stage 2", "title": "Pose Estimation",
            "items": ["• MediaPipe Pose Framework", "• 33 2D/3D Body Landmarks", "• Sub-pixel Landmark Smoothing", "• Hips, Knees, Ankles, Torso"],
            "color": "#3498db", "x": 3.6
        },
        {
            "step": "Stage 3", "title": "Kinematic Extraction",
            "items": ["• Total TUG & Transition Times", "• Walking Speed & Cadence", "• Stride Length Variability", "• Trunk Sway & Asymmetry Index", "• Dual-Task Cost (DTC %)"],
            "color": "#9b59b6", "x": 6.4
        },
        {
            "step": "Stage 4", "title": "ML Classification",
            "items": ["• Feature Scaling & Normalization", "• Support Vector Machine (SVM)", "• Random Forest Classifier", "• XGBoost Gradient Boosting"],
            "color": "#e67e22", "x": 9.2
        },
        {
            "step": "Stage 5", "title": "Clinical Output",
            "items": ["• Healthy vs. Impairment Flag", "• Probability / Confidence Score", "• SHAP Feature Attribution", "• Interactive Web Dashboard", "• Downloadable PDF Report"],
            "color": "#e74c3c", "x": 12.0
        }
    ]

    box_w = 2.4
    box_h = 4.8
    y = 1.0

    for i, s in enumerate(stages):
        # Card body
        rect = FancyBboxPatch((s["x"], y), box_w, box_h,
                              boxstyle="round,pad=0.1,rounding_size=0.2",
                              facecolor="#f8fafc", edgecolor=s["color"], linewidth=2.5,
                              alpha=0.95)
        ax.add_patch(rect)
        
        # Header banner
        header = FancyBboxPatch((s["x"], y + box_h - 0.85), box_w, 0.85,
                                boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor=s["color"], edgecolor=s["color"])
        ax.add_patch(header)
        
        ax.text(s["x"] + box_w/2, y + box_h - 0.35, s["step"],
                fontsize=11, fontweight="bold", color="white", ha="center", va="center")
        ax.text(s["x"] + box_w/2, y + box_h - 0.65, s["title"],
                fontsize=10.5, fontweight="bold", color="#ffffff", ha="center", va="center")
        
        # Bullet items
        bullet_y = y + box_h - 1.3
        for item in s["items"]:
            ax.text(s["x"] + 0.15, bullet_y, item,
                    fontsize=9, color="#2d3748", ha="left", va="top")
            bullet_y -= 0.65

        # Connection Arrow
        if i < len(stages) - 1:
            ax.annotate("", xy=(s["x"] + box_w + 0.35, y + box_h/2),
                        xytext=(s["x"] + box_w + 0.05, y + box_h/2),
                        arrowprops=dict(facecolor="#4a5568", edgecolor="#4a5568", width=2.5, headwidth=8))

    # Bottom SDG Badge
    ax.text(7.5, 0.45, "Aligned with UN Sustainable Development Goals: SDG 3 (Good Health & Well-being) & SDG 9 (Industry, Innovation & Infrastructure)",
            fontsize=9.5, color="#1a365d", ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ebf8ff", edgecolor="#bee3f8"))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Created: {output_path}")


if __name__ == "__main__":
    create_tug_phases_diagram()
    create_system_architecture_diagram()
