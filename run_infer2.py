import os
import sys
import cv2
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from ultralytics import YOLO
from models.shopformer_transformer import Shopformer
from utils.pose_tokenizer import normalize_keypoints

# === Pose Extraction without Batching ===
def extract_pose_sequence_no_batch(video_path):
    model = YOLO("yolov8m-pose.pt")
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
            frames.append(np.zeros((17, 2)))

    cap.release()
    return normalize_keypoints(frames, (H, W)), (H, W)

# === Pose Extraction with Batching ===
def extract_pose_sequence_batch(video_path, batch_size=16):
    model = YOLO("yolov8m-pose.pt")
    cap = cv2.VideoCapture(video_path)
    H, W = None, None
    batch = []
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if H is None or W is None:
            H, W = frame.shape[:2]
        batch.append(frame)

        if len(batch) == batch_size:
            results = model.predict(batch, verbose=False)
            for r in results:
                if r.keypoints is not None and r.keypoints.xy.numel() > 0:
                    frames.append(r.keypoints.xy[0].cpu().numpy())
                else:
                    frames.append(np.zeros((17, 2)))
            batch = []

    # Process remaining frames
    if batch:
        results = model.predict(batch, verbose=False)
        for r in results:
            if r.keypoints is not None and r.keypoints.xy.numel() > 0:
                frames.append(r.keypoints.xy[0].cpu().numpy())
            else:
                frames.append(np.zeros((17, 2)))

    cap.release()
    return normalize_keypoints(frames, (H, W)), (H, W)

# === Optional Ground Truth CSV Loader ===
def load_ground_truth_csv(csv_path, fps):
    df = pd.read_csv(csv_path, quotechar='"', escapechar='\\')
    frame_labels = {}

    for _, row in df.iterrows():
        try:
            start_time = float(row["temporal_segment_start"])
            end_time = float(row["temporal_segment_end"])
            metadata_str = row["metadata"]
            metadata = json.loads(metadata_str.strip().strip('"'))
            label = metadata["TEMPORAL_SEGMENTS"].lower()

            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            for f in range(start_frame, end_frame + 1):
                frame_labels[f] = 1 if label == "shoplifting" else 0

        except Exception as e:
            print(f"⚠️ Skipping row due to parse error: {e}")

    return frame_labels

# === Metrics ===
def compute_metrics(frame_labels_gt, frame_scores, threshold):
    frame_ids = sorted(frame_scores.keys())
    y_pred = [1 if frame_scores[f] > threshold else 0 for f in frame_ids]
    y_true = [frame_labels_gt.get(f, 0) for f in frame_ids]
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

# === Main Pipeline ===
def main(video_path, gt_csv_path=None, use_batch=True):
    os.makedirs("outputs", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Shopformer().to(device)
    model.eval()

    print("🔍 Extracting pose keypoints...")
    if use_batch:
        pose_seq, (H, W) = extract_pose_sequence_batch(video_path)
    else:
        pose_seq, (H, W) = extract_pose_sequence_no_batch(video_path)

    pose_seq = pose_seq.unsqueeze(0).to(device)

    print("⚙️ Running Shopformer inference...")
    with torch.no_grad():
        output = model(pose_seq)
        mse = torch.mean((pose_seq - output) ** 2, dim=-1).squeeze(0).cpu().numpy()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    anomaly_writer = cv2.VideoWriter("outputs/anomaly_only_video.mp4", fourcc, fps, (W, H))
    normal_writer = cv2.VideoWriter("outputs/normal_only_video.mp4", fourcc, fps, (W, H))

    threshold = 0.4
    frame_idx = 0
    anomaly_count, normal_count = 0, 0
    frame_scores = {}

    print("\n🎬 Writing video outputs...")
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

        frame_scores[frame_idx] = score
        frame_idx += 1

    cap.release()
    anomaly_writer.release()
    normal_writer.release()

    print(f"\n✅ Saved {anomaly_count} anomalous frames → outputs/anomaly_only_video.mp4")
    print(f"✅ Saved {normal_count} normal frames → outputs/normal_only_video.mp4")
    print(f"📊 Anomaly Ratio: {anomaly_count}/{frame_idx} = {100 * anomaly_count / frame_idx:.2f}%")

    # === Optional Metrics ===
    if gt_csv_path:
        print("\n📋 Evaluating metrics...")
        frame_labels_gt = load_ground_truth_csv(gt_csv_path, fps)
        precision, recall, f1 = compute_metrics(frame_labels_gt, frame_scores, threshold)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
    else:
        print("\n⚠️ No ground truth CSV provided. Skipping metrics computation.")

    # === Plot ===
    plt.figure(figsize=(10, 4))
    plt.plot(mse, label="Anomaly Score (MSE)")
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    plt.title("Frame-wise Anomaly Score")
    plt.xlabel("Frame")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/normality_score_plot.png")
    print("📈 Saved plot → outputs/normality_score_plot.png")

# === Entry Point ===
if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("Usage: python run_infer2.py <path_to_video> [optional_ground_truth_csv]")
        sys.exit(1)

    video_path = sys.argv[1]
    gt_csv_path = sys.argv[2] if len(sys.argv) == 3 else None

    # Toggle batching here
    use_batch = True  # Set to False to disable batching

    main(video_path, gt_csv_path, use_batch)
