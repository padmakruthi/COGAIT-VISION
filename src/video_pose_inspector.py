"""
COGAIT-VISION: Frame-by-Frame Pose & Skeletal Trajectory Inspector
Extracts detailed joint angles (knee, hip, trunk) and landmark trajectories for deep clinical analysis.
Clinical AI Research Initiative
"""

import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from src.pose_detector import PoseDetector


def calculate_angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate angle ABC at vertex B in degrees."""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def extract_full_pose_trajectories(video_path: str, max_frames: int = 300) -> Dict[str, Any]:
    """
    Analyzes video frame-by-frame, extracting:
      - Landmark coordinates per frame
      - Left & Right Knee angles
      - Left & Right Hip angles
      - Trunk Sway angles
      - Hip height (for sit-to-stand segmentation)
      - Sample annotated frames for UI scrubbing
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 24.0

    detector = PoseDetector()

    frames_data = []
    annotated_frames = []
    frame_indices = []

    hip_y_trajectory = []
    left_knee_angles = []
    right_knee_angles = []
    trunk_angles = []
    shoulder_widths = []
    left_ankle_y = []
    right_ankle_y = []

    idx = 0
    while cap.isOpened() and idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, landmarks = detector.process_frame(frame, draw=True)

        if landmarks is not None and 23 in landmarks and 24 in landmarks:
            # Hip midpoint
            mid_hip = (landmarks[23][:2] + landmarks[24][:2]) / 2.0
            mid_shoulder = (landmarks[11][:2] + landmarks[12][:2]) / 2.0
            
            # Trunk Angle (relative to vertical)
            trunk_vec = mid_shoulder - mid_hip
            t_angle = float(np.degrees(np.arctan2(trunk_vec[0], -trunk_vec[1])))

            # Left Knee Angle (Hip 23, Knee 25, Ankle 27)
            lk_angle = calculate_angle_2d(landmarks[23][:2], landmarks[25][:2], landmarks[27][:2])
            # Right Knee Angle (Hip 24, Knee 26, Ankle 28)
            rk_angle = calculate_angle_2d(landmarks[24][:2], landmarks[26][:2], landmarks[28][:2])

            # Shoulder separation
            sh_w = float(abs(landmarks[11][0] - landmarks[12][0]))

            hip_y_trajectory.append(float(mid_hip[1]))
            left_knee_angles.append(float(lk_angle))
            right_knee_angles.append(float(rk_angle))
            trunk_angles.append(float(t_angle))
            shoulder_widths.append(sh_w)
            left_ankle_y.append(float(landmarks[27][1]))
            right_ankle_y.append(float(landmarks[28][1]))
            frame_indices.append(idx)

            # Store thumbnail every 5 frames for scrubbing
            if idx % 5 == 0:
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                annotated_frames.append((idx, annotated_rgb, landmarks))

        idx += 1

    cap.release()
    detector.close()

    time_seconds = [f / fps for f in frame_indices]

    df_trajectories = pd.DataFrame({
        "frame": frame_indices,
        "time_s": [round(t, 2) for t in time_seconds],
        "hip_y": hip_y_trajectory,
        "left_knee_angle": left_knee_angles,
        "right_knee_angle": right_knee_angles,
        "trunk_sway_angle": trunk_angles,
        "shoulder_width": shoulder_widths,
        "left_ankle_y": left_ankle_y,
        "right_ankle_y": right_ankle_y,
    })

    return {
        "df_trajectories": df_trajectories,
        "thumbnails": annotated_frames,
        "fps": fps,
        "total_frames_analyzed": len(frame_indices),
    }
