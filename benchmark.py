import torch
import time
import sys
from models.shopformer_transformer import Shopformer
from models.shopformer_v2 import ShopformerV2
from run_infer_v2 import extract_poses

def benchmark(video_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    print("Extracting poses...")
    pose_seq, _ = extract_poses(video_path)
    n = len(pose_seq)
    print(f"Frames: {n}\n")
    
    print("-" * 60)
    print("V1: Simple Transformer")
    print("-" * 60)
    
    m1 = Shopformer().to(device)
    m1.eval()
    p1 = sum(p.numel() for p in m1.parameters())
    
    x1 = pose_seq.unsqueeze(0).to(device)
    with torch.no_grad():
        _ = m1(x1)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    t0 = time.time()
    with torch.no_grad():
        for _ in range(5):
            _ = m1(x1)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = (time.time() - t0) / 5
    
    print(f"Parameters: {p1:,}")
    print(f"Time: {t1*1000:.0f}ms ({n/t1:.0f} fps)")
    
    print("\n" + "-" * 60)
    print("V2: Graph + Transformer")
    print("-" * 60)
    
    m2 = ShopformerV2().to(device)
    m2.eval()
    p2 = sum(p.numel() for p in m2.parameters())
    
    x2 = pose_seq.reshape(1, n, 17, 2).to(device)
    with torch.no_grad():
        _ = m2(x2)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    t0 = time.time()
    with torch.no_grad():
        for _ in range(5):
            _ = m2(x2)
    if device == "cuda":
        torch.cuda.synchronize()
    t2 = (time.time() - t0) / 5
    
    print(f"Parameters: {p2:,}")
    print(f"Time: {t2*1000:.0f}ms ({n/t2:.0f} fps)")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"V1: {p1:,} params, {n/t1:.0f} fps")
    print(f"V2: {p2:,} params, {n/t2:.0f} fps ({p2/p1:.1f}x params, {t2/t1:.1f}x slower)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python benchmark.py <video>")
        sys.exit(1)
    benchmark(sys.argv[1])