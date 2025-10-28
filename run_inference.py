import torch
from models.shopformer_transformer import Shopformer
from utils.yolov8_pose_extractor import extract_pose_sequence
import sys

def main(video_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Shopformer().to(device)
    model.eval()

    pose_seq = extract_pose_sequence(video_path).unsqueeze(0).to(device)
    with torch.no_grad():
        reconstructed = model(pose_seq)
        mse = torch.mean((pose_seq - reconstructed)**2, dim=-1).squeeze(0).cpu().numpy()
        for i, score in enumerate(mse):
            print(f"Frame {i}: Score = {score:.4f}")

if __name__ == "__main__":
    main(sys.argv[1])
