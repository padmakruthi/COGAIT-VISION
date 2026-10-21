"""
COGAIT-VISION: MediaPipe Pose Landmark Detection Module
Draws glowing joints, color-coded limbs, and live joint angles (Knee, Hip, Trunk).
Department of Computer Science and Engineering, IARE
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, Tuple, Dict, Any, List


# Color Palette (BGR)
COLOR_TORSO = (255, 200, 0)      # Cyan / Gold
COLOR_LEFT_LIMB = (50, 220, 50)   # Vivid Green
COLOR_RIGHT_LIMB = (50, 50, 240)  # Vivid Red
COLOR_ARMS = (240, 180, 50)       # Cyan Blue
COLOR_JOINT = (255, 255, 255)     # Bright White
COLOR_JOINT_BORDER = (0, 0, 0)


def calculate_angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate angle ABC at vertex B in degrees."""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 180.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class PoseDetector:
    """
    Robust pose detection engine utilizing MediaPipe Tasks PoseLandmarker.
    Renders high-visibility skeletons, joint angles, and clinical posture annotations.
    """

    def __init__(self, model_path: str = "models/pose_landmarker_lite.task", **kwargs):
        self.model_path = model_path
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        if not os.path.exists(self.model_path):
            print(f"[Warning] Pose model file not found at {self.model_path}.")
            return

        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.45,
                min_pose_presence_confidence=0.45,
                min_tracking_confidence=0.45,
                output_segmentation_masks=False,
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
            print("[Info] MediaPipe PoseLandmarker initialized successfully.")
        except Exception as e:
            print(f"[Error] Failed to initialize PoseLandmarker: {e}")
            self.detector = None

    def process_frame(
        self, frame: np.ndarray, draw: bool = True, show_angles: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict[int, np.ndarray]]]:
        """
        Processes a BGR video frame and returns:
            - annotated_frame with colorful skeleton and joint angle labels
            - landmarks_dict: {idx: np.array([x, y, z, visibility])}
        """
        h, w, _ = frame.shape
        annotated_frame = frame.copy()
        landmarks_dict = None

        if self.detector is None:
            return annotated_frame, None

        # Convert to RGB and wrap in MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(mp_image)

        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            pose_lms = detection_result.pose_landmarks[0]
            landmarks_dict = {}

            # Map landmark indices
            for idx, lm in enumerate(pose_lms):
                landmarks_dict[idx] = np.array([lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)])

            if draw:
                # 1. Draw Bones with Distinct Segment Colors
                limb_groups = [
                    # Torso
                    ([(11, 12), (11, 23), (12, 24), (23, 24)], COLOR_TORSO, 4),
                    # Left Leg
                    ([(23, 25), (25, 27), (27, 29), (29, 31)], COLOR_LEFT_LIMB, 4),
                    # Right Leg
                    ([(24, 26), (26, 28), (28, 30), (30, 32)], COLOR_RIGHT_LIMB, 4),
                    # Left Arm
                    ([(11, 13), (13, 15)], COLOR_ARMS, 3),
                    # Right Arm
                    ([(12, 14), (14, 16)], COLOR_ARMS, 3),
                ]

                for connections, color, thickness in limb_groups:
                    for p1, p2 in connections:
                        if p1 in landmarks_dict and p2 in landmarks_dict:
                            pt1 = (int(landmarks_dict[p1][0] * w), int(landmarks_dict[p1][1] * h))
                            pt2 = (int(landmarks_dict[p2][0] * w), int(landmarks_dict[p2][1] * h))
                            cv2.line(annotated_frame, pt1, pt2, color, thickness, cv2.LINE_AA)

                # 2. Draw Glowing Joint Nodes
                key_joints = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
                for idx in key_joints:
                    if idx in landmarks_dict:
                        px = int(landmarks_dict[idx][0] * w)
                        py = int(landmarks_dict[idx][1] * h)
                        # Outer ring
                        cv2.circle(annotated_frame, (px, py), 7, (0, 0, 0), -1, cv2.LINE_AA)
                        # Inner glow
                        node_color = (0, 255, 255) if idx in [23, 24, 25, 26, 27, 28] else (255, 255, 255)
                        cv2.circle(annotated_frame, (px, py), 5, node_color, -1, cv2.LINE_AA)

                # 3. Draw Joint Angle Annotations Directly on Joints
                if show_angles and 23 in landmarks_dict and 25 in landmarks_dict and 27 in landmarks_dict:
                    # Left Knee Angle
                    l_knee_angle = calculate_angle_2d(landmarks_dict[23][:2], landmarks_dict[25][:2], landmarks_dict[27][:2])
                    lk_x, lk_y = int(landmarks_dict[25][0] * w), int(landmarks_dict[25][1] * h)
                    cv2.putText(annotated_frame, f"{int(l_knee_angle)} deg", (lk_x + 10, lk_y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 255, 50), 2, cv2.LINE_AA)

                if show_angles and 24 in landmarks_dict and 26 in landmarks_dict and 28 in landmarks_dict:
                    # Right Knee Angle
                    r_knee_angle = calculate_angle_2d(landmarks_dict[24][:2], landmarks_dict[26][:2], landmarks_dict[28][:2])
                    rk_x, rk_y = int(landmarks_dict[26][0] * w), int(landmarks_dict[26][1] * h)
                    cv2.putText(annotated_frame, f"{int(r_knee_angle)} deg", (rk_x - 70, rk_y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 80, 255), 2, cv2.LINE_AA)

                if show_angles and 11 in landmarks_dict and 12 in landmarks_dict and 23 in landmarks_dict and 24 in landmarks_dict:
                    mid_shoulder = (landmarks_dict[11][:2] + landmarks_dict[12][:2]) / 2.0
                    mid_hip = (landmarks_dict[23][:2] + landmarks_dict[24][:2]) / 2.0
                    trunk_vec = mid_shoulder - mid_hip
                    trunk_sway = float(np.degrees(np.arctan2(trunk_vec[0], -trunk_vec[1])))
                    t_x, t_y = int(mid_shoulder[0] * w), int(mid_shoulder[1] * h)
                    cv2.putText(annotated_frame, f"Trunk: {abs(trunk_sway):.1f} deg", (t_x - 45, t_y - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 230, 0), 2, cv2.LINE_AA)

        return annotated_frame, landmarks_dict

    def close(self):
        if self.detector is not None:
            self.detector.close()
