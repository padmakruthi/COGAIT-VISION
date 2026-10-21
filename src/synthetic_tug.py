"""
COGAIT-VISION: Synthetic TUG Landmark & Video Generator
Generates realistic 3D skeletal landmark trajectories and demo video for testing without physical patients.
Department of Computer Science and Engineering, IARE
Team: B P S Kruthi, Ch. Karthika, A. Laxmi Sahithi | Guide: Ms. B K Aishwarya
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple


def generate_simulated_tug_landmarks(
    condition: str = "healthy", fps: float = 30.0
) -> List[Dict[int, np.ndarray]]:
    """
    Generates a realistic sequence of 33 MediaPipe-format landmarks 
    simulating the 5 phases of the Timed Up and Go (TUG) test.
    
    condition: 'healthy' or 'impaired'
    """
    if condition == "healthy":
        duration = 9.5
        sway_factor = 0.008
        sit_duration = 1.5
        walk_speed = 1.0
    else:
        duration = 17.5
        sway_factor = 0.028
        sit_duration = 3.2
        walk_speed = 0.55

    num_frames = int(duration * fps)
    landmarks_seq = []

    # Baseline landmark normalized coordinates
    # Hips around y=0.55 (standing) to 0.75 (sitting)
    # Ankles around y=0.90
    # Shoulders around y=0.30
    # Nose around y=0.18

    for t in range(num_frames):
        progress = t / num_frames
        lm = {}

        # 5 phases:
        # 0.0 to 0.15: Sit-to-Stand
        # 0.15 to 0.45: Walk forward
        # 0.45 to 0.55: 180 turn
        # 0.55 to 0.85: Walk back
        # 0.85 to 1.00: Stand-to-Sit

        if progress < 0.15:
            # Sit to stand
            phase_prog = progress / 0.15
            hip_y = 0.72 - 0.18 * phase_prog
            shoulder_y = 0.48 - 0.18 * phase_prog
            x_offset = 0.5
            z_offset = 0.0
        elif progress < 0.45:
            # Walk forward
            phase_prog = (progress - 0.15) / 0.30
            hip_y = 0.54 + 0.01 * np.sin(phase_prog * 20)
            shoulder_y = 0.30 + 0.01 * np.sin(phase_prog * 20)
            x_offset = 0.5 + np.sin(phase_prog * 10) * sway_factor
            z_offset = -0.3 * phase_prog
        elif progress < 0.55:
            # Turn
            phase_prog = (progress - 0.45) / 0.10
            hip_y = 0.54
            shoulder_y = 0.30
            x_offset = 0.5 + 0.03 * np.sin(phase_prog * np.pi)
            z_offset = -0.3
        elif progress < 0.85:
            # Walk back
            phase_prog = (progress - 0.55) / 0.30
            hip_y = 0.54 + 0.01 * np.sin(phase_prog * 20)
            shoulder_y = 0.30 + 0.01 * np.sin(phase_prog * 20)
            x_offset = 0.5 + np.sin(phase_prog * 10) * sway_factor
            z_offset = -0.3 * (1.0 - phase_prog)
        else:
            # Stand to sit
            phase_prog = (progress - 0.85) / 0.15
            hip_y = 0.54 + 0.18 * phase_prog
            shoulder_y = 0.30 + 0.18 * phase_prog
            x_offset = 0.5
            z_offset = 0.0

        # Ankle stride alternation
        gait_cycle = np.sin(progress * num_frames * 0.4)
        left_ankle_y = 0.88 + 0.04 * max(0.0, gait_cycle)
        right_ankle_y = 0.88 + 0.04 * max(0.0, -gait_cycle)

        # Build 33 landmarks
        lm[0] = np.array([x_offset, shoulder_y - 0.12, z_offset, 1.0])  # Nose
        lm[11] = np.array([x_offset - 0.09, shoulder_y, z_offset, 1.0])  # Left Shoulder
        lm[12] = np.array([x_offset + 0.09, shoulder_y, z_offset, 1.0])  # Right Shoulder
        lm[23] = np.array([x_offset - 0.06, hip_y, z_offset, 1.0])  # Left Hip
        lm[24] = np.array([x_offset + 0.06, hip_y, z_offset, 1.0])  # Right Hip
        lm[25] = np.array([x_offset - 0.06, (hip_y + left_ankle_y) / 2.0, z_offset, 1.0])  # Left Knee
        lm[26] = np.array([x_offset + 0.06, (hip_y + right_ankle_y) / 2.0, z_offset, 1.0])  # Right Knee
        lm[27] = np.array([x_offset - 0.06, left_ankle_y, z_offset, 1.0])  # Left Ankle
        lm[28] = np.array([x_offset + 0.06, right_ankle_y, z_offset, 1.0])  # Right Ankle
        lm[29] = np.array([x_offset - 0.06, left_ankle_y + 0.01, z_offset, 1.0])  # Left Heel
        lm[30] = np.array([x_offset + 0.06, right_ankle_y + 0.01, z_offset, 1.0])  # Right Heel
        lm[31] = np.array([x_offset - 0.06, left_ankle_y + 0.01, z_offset + 0.05, 1.0])
        lm[32] = np.array([x_offset + 0.06, right_ankle_y + 0.01, z_offset + 0.05, 1.0])

        landmarks_seq.append(lm)

    return landmarks_seq


def render_synthetic_tug_video(
    output_filename: str = "data/sample_tug_healthy.mp4", condition: str = "healthy", fps: int = 30
):
    """Creates a sample MP4 video visualizing the TUG motion for demonstration."""
    import os
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    landmarks_seq = generate_simulated_tug_landmarks(condition=condition, fps=fps)
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    colors = (0, 255, 120) if condition == "healthy" else (50, 100, 255)

    for i, lm in enumerate(landmarks_seq):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (30, 30, 35)

        # Draw floor line
        cv2.line(frame, (50, int(height * 0.92)), (width - 50, int(height * 0.92)), (70, 70, 80), 2)
        
        # Draw Chair
        cv2.rectangle(frame, (100, int(height * 0.65)), (160, int(height * 0.92)), (60, 60, 90), -1)
        cv2.putText(frame, "Chair", (105, int(height * 0.63)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Draw 3m Marker
        cv2.line(frame, (500, int(height * 0.85)), (500, int(height * 0.92)), (0, 200, 255), 3)
        cv2.putText(frame, "3m Turn", (470, int(height * 0.83)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # Draw skeleton
        connections = [(11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]
        for p1, p2 in connections:
            pt1 = (int(lm[p1][0] * width), int(lm[p1][1] * height))
            pt2 = (int(lm[p2][0] * width), int(lm[p2][1] * height))
            cv2.line(frame, pt1, pt2, colors, 3)

        for idx in [0, 11, 12, 23, 24, 25, 26, 27, 28]:
            pt = (int(lm[idx][0] * width), int(lm[idx][1] * height))
            cv2.circle(frame, pt, 5, (255, 255, 255), -1)

        # Info overlay
        phase_str = f"Condition: {condition.upper()} | Frame: {i}/{len(landmarks_seq)}"
        cv2.putText(frame, phase_str, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
        out.write(frame)

    out.release()
    print(f"Generated synthetic video: {output_filename}")


if __name__ == "__main__":
    render_synthetic_tug_video("data/sample_tug_healthy.mp4", "healthy")
    render_synthetic_tug_video("data/sample_tug_impaired.mp4", "impaired")
