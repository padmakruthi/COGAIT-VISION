"""
Generate Web-Compatible H.264 Annotated Videos with Full Skeleton and Joint Tracking
"""

import os
import sys
import subprocess
import cv2
import imageio_ffmpeg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pose_detector import PoseDetector


def render_h264_annotated_video(input_path: str, output_path: str, condition_title: str):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    detector = PoseDetector()

    # Create temporary raw annotated video
    temp_avi = output_path.replace(".mp4", "_temp.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(temp_avi, fourcc, fps, (w, h))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process skeleton with angles
        annotated_frame, landmarks = detector.process_frame(frame, draw=True, show_angles=True)

        # Header overlay
        header_color = (40, 160, 40) if "Normal" in condition_title else (40, 40, 200)
        cv2.rectangle(annotated_frame, (20, 20), (540, 85), (20, 20, 30), -1)
        cv2.rectangle(annotated_frame, (20, 20), (540, 85), header_color, 2)
        cv2.putText(annotated_frame, f"TUG TRACKER: {condition_title}", (35, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        
        status_sub = "Joint Tracking: 33 Keypoints Active"
        cv2.putText(annotated_frame, status_sub, (35, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 240, 255), 1, cv2.LINE_AA)

        out.write(annotated_frame)
        frame_idx += 1

    cap.release()
    out.release()
    detector.close()

    # Convert to web-standard H.264 using bundled ffmpeg
    cmd = [
        ffmpeg_exe, "-y",
        "-i", temp_avi,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(temp_avi):
        os.remove(temp_avi)

    print(f"Rendered H.264 web video: {output_path} ({frame_idx} frames)")


if __name__ == "__main__":
    os.makedirs("data/videos", exist_ok=True)
    render_h264_annotated_video("data/videos/normal walk.mp4", "data/videos/web_annotated_normal.mp4", "Normal Walk (Healthy)")
    render_h264_annotated_video("data/videos/abnormal.mp4", "data/videos/web_annotated_abnormal.mp4", "Abnormal Walk (Alzheimer Risk)")
