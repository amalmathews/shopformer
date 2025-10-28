import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from models.shopformer_transformer import Shopformer
from utils.pose_tokenizer import normalize_keypoints

def extract_pose_sequence(video_path):
    model = YOLO("yolov8m-pose.pt")  # Automatically downloads model
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
        else:
            frames.append(np.zeros((17, 2)))  # Default empty keypoints

    cap.release()
    return normalize_keypoints(frames, (H, W)), (H, W)

def main(video_path):
    # Ensure output folder
    os.makedirs("outputs", exist_ok=True)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Shopformer().to(device)
    model.eval()

    # Extract pose
    pose_seq, (H, W) = extract_pose_sequence(video_path)
    pose_seq = pose_seq.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(pose_seq)
        mse = torch.mean((pose_seq - output) ** 2, dim=-1).squeeze(0).cpu().numpy()

    # Video output setup
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    anomaly_writer = cv2.VideoWriter("outputs/anomaly_only_video.mp4", fourcc, fps, (W, H))
    normal_writer = cv2.VideoWriter("outputs/normal_only_video.mp4", fourcc, fps, (W, H))

    threshold = 0.4
    frame_idx = 0
    anomaly_count = 0
    normal_count = 0

    print("🎬 Writing video outputs...\n")
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(mse):
            break

        score = mse[frame_idx]
        label = "Shoplifting" if score > threshold else "Normal"
        color = (0, 0, 255) if score > threshold else (0, 255, 0)

        cv2.putText(frame, f"{label} ({score:.2f})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        cv2.rectangle(frame, (10, 10), (W - 10, H - 10), color, 4)

        if score > threshold:
            anomaly_writer.write(frame)
            anomaly_count += 1
        else:
            normal_writer.write(frame)
            normal_count += 1

        print(f"Frame {frame_idx:03d}: Score = {score:.4f} → {label}")
        frame_idx += 1

    cap.release()
    anomaly_writer.release()
    normal_writer.release()

    print(f"\n✅ Saved {anomaly_count} anomalous frames → outputs/anomaly_only_video.mp4")
    print(f"✅ Saved {normal_count} normal frames → outputs/normal_only_video.mp4")
    print(f"📊 Anomaly Ratio: {anomaly_count}/{frame_idx} = {100 * anomaly_count / frame_idx:.2f}%")

    # Plot MSE anomaly scores
    plt.figure(figsize=(10, 4))
    plt.plot(mse, label="Anomaly Score (MSE)")
    plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold = 0.4')
    plt.title("Frame-wise Anomaly Score")
    plt.xlabel("Frame")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/normality_score_plot.png")
    print("📈 Saved plot → outputs/normality_score_plot.png")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_infer2.py <path_to_video>")
        sys.exit(1)
    main(sys.argv[1])
