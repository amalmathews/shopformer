from ultralytics import YOLO
import cv2
import numpy as np
from .pose_tokenizer import normalize_keypoints

def extract_pose_sequence(video_path):
    model = YOLO('yolov8m-pose.pt')  # download automatically
    cap = cv2.VideoCapture(video_path)
    frames = []
    H, W = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W = frame.shape[:2]
        results = model.predict(frame, verbose=False)
        kp = results[0].keypoints.xy.cpu().numpy()
        if kp.shape[0] > 0:
            frames.append(kp[0])  # First person
    cap.release()

    return normalize_keypoints(frames, (H, W))
