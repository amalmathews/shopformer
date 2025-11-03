import os
import sys
import cv2
import torch
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from ultralytics import YOLO
from models.shopformer_transformer import Shopformer
from models.shopformer_v2 import ShopformerV2
from utils.pose_tokenizer import normalize_keypoints

def extract_poses(video_path, batch_size=16):
    model = YOLO("yolov8m-pose.pt")
    cap = cv2.VideoCapture(video_path)
    H, W = None, None
    batch = []
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if H is None:
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

    if batch:
        results = model.predict(batch, verbose=False)
        for r in results:
            if r.keypoints is not None and r.keypoints.xy.numel() > 0:
                frames.append(r.keypoints.xy[0].cpu().numpy())
            else:
                frames.append(np.zeros((17, 2)))

    cap.release()
    return normalize_keypoints(frames, (H, W)), (H, W)


def load_labels(csv_path, fps):
    df = pd.read_csv(csv_path, quotechar='"', escapechar='\\')
    labels = {}
    for _, row in df.iterrows():
        try:
            start = int(float(row["temporal_segment_start"]) * fps)
            end = int(float(row["temporal_segment_end"]) * fps)
            meta = json.loads(row["metadata"].strip().strip('"'))
            is_theft = 1 if meta["TEMPORAL_SEGMENTS"].lower() == "shoplifting" else 0
            for f in range(start, end + 1):
                labels[f] = is_theft
        except:
            continue
    return labels


def run_inference(video_path, csv_path=None, model_version='v2'):
    os.makedirs("outputs", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_version == 'v1':
        print("Loading V1 model...")
        model = Shopformer(input_dim=34, d_model=128, nhead=4, num_layers=2).to(device)
    else:
        print("Loading V2 model...")
        model = ShopformerV2().to(device)
        info = model.get_model_info()
        print(f"Parameters: {info['total_parameters']:,}")
    
    model.eval()

    print("Extracting poses...")
    t0 = time.time()
    pose_seq, (H, W) = extract_poses(video_path)
    pose_time = time.time() - t0
    num_frames = len(pose_seq)
    
    print(f"Extracted {num_frames} frames in {pose_time:.1f}s ({num_frames/pose_time:.1f} fps)")
    
    if model_version == 'v1':
        pose_input = pose_seq.unsqueeze(0).to(device)
    else:
        pose_input = pose_seq.reshape(1, num_frames, 17, 2).to(device)

    print("Running inference...")
    t0 = time.time()
    
    with torch.no_grad():
        output = model(pose_input)
        if model_version == 'v1':
            mse = torch.mean((pose_input - output) ** 2, dim=-1).squeeze(0).cpu().numpy()
        else:
            mse = torch.mean((pose_input - output) ** 2, dim=(2, 3)).squeeze(0).cpu().numpy()
    
    inf_time = time.time() - t0
    print(f"Inference: {inf_time:.1f}s ({num_frames/inf_time:.1f} fps)")
    print(f"Total: {pose_time + inf_time:.1f}s")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    anom_writer = cv2.VideoWriter("outputs/anomaly.mp4", fourcc, fps, (W, H))
    norm_writer = cv2.VideoWriter("outputs/normal.mp4", fourcc, fps, (W, H))

    threshold = 0.6
    frame_idx = 0
    anom_cnt = 0
    norm_cnt = 0
    scores = {}

    print("Writing outputs...")
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(mse):
            break

        score = mse[frame_idx]
        is_anom = score > threshold
        label = "Theft" if is_anom else "Normal"
        color = (0, 0, 255) if is_anom else (0, 255, 0)

        cv2.putText(frame, f"{label} {score:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        if is_anom:
            anom_writer.write(frame)
            anom_cnt += 1
        else:
            norm_writer.write(frame)
            norm_cnt += 1

        scores[frame_idx] = score
        frame_idx += 1

    cap.release()
    anom_writer.release()
    norm_writer.release()

    print(f"\nResults:")
    print(f"  Anomaly frames: {anom_cnt}")
    print(f"  Normal frames: {norm_cnt}")
    print(f"  Ratio: {100*anom_cnt/frame_idx:.1f}%")

    if csv_path:
        print("\nEvaluating...")
        labels = load_labels(csv_path, fps)
        frame_ids = sorted(scores.keys())
        y_pred = [1 if scores[f] > threshold else 0 for f in frame_ids]
        y_true = [labels.get(f, 0) for f in frame_ids]
        
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"  Precision: {p:.3f}")
        print(f"  Recall: {r:.3f}")
        print(f"  F1: {f1:.3f}")

    plt.figure(figsize=(10, 3))
    plt.plot(mse, linewidth=0.8)
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=0.8)
    plt.xlabel("Frame")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("outputs/scores.png", dpi=100)
    print("\nSaved outputs/scores.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_infer_v2.py <video> [csv] [--model v1|v2]")
        sys.exit(1)

    video = sys.argv[1]
    csv = None
    version = 'v2'
    
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == '--model' and i + 1 < len(sys.argv):
            version = sys.argv[i + 1]
        elif not arg.startswith('--') and arg not in ['v1', 'v2']:
            csv = arg
    
    run_inference(video, csv, version)