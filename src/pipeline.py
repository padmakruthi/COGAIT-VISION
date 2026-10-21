"""
COGAIT-VISION: End-to-End TUG Video Processing Pipeline
Integrates video frame capture, MediaPipe skeletal tracking, kinematic calculation, and ML scoring.
Department of Computer Science and Engineering, IARE
Team: B P S Kruthi, Ch. Karthika, A. Laxmi Sahithi | Guide: Ms. B K Aishwarya
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, Callable
from src.pose_detector import PoseDetector
from src.kinematic_features import TUGKinematicFeatureExtractor
from src.classifier import TUGClassifier


class TUGAnalysisPipeline:
    """
    Complete end-to-end pipeline for processing monocular TUG video or webcam feeds.
    """

    def __init__(self, models_dir: str = "models", fps: float = 30.0):
        self.fps = fps
        self.detector = PoseDetector()
        self.extractor = TUGKinematicFeatureExtractor(fps=fps)
        self.classifier = TUGClassifier(models_dir=models_dir)

    def process_video(
        self,
        video_path: str,
        output_annotated_path: Optional[str] = None,
        dual_task: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Processes a full TUG video file and extracts diagnostic predictions.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = self.fps

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_annotated_path:
            os.makedirs(os.path.dirname(output_annotated_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_annotated_path, fourcc, fps, (width, height))

        landmark_seq = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, landmarks = self.detector.process_frame(frame, draw=True)
            if landmarks:
                landmark_seq.append(landmarks)

            if writer:
                writer.write(annotated_frame)

            frame_idx += 1
            if progress_callback and total_frames > 0:
                progress_callback(min(1.0, frame_idx / total_frames))

        cap.release()
        if writer:
            writer.release()

        # Extract kinematics
        features = self.extractor.extract_features_from_landmark_sequence(
            landmark_seq, dual_task_present=dual_task
        )

        # Classify with ML models
        results = self.classifier.predict(features)
        results["total_frames_processed"] = frame_idx
        results["detected_landmark_frames"] = len(landmark_seq)
        results["annotated_video_path"] = output_annotated_path

        return results
