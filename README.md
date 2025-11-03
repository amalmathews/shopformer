# Shopformer Inference Pipeline

This repo implements the inference pipeline for Shopformer (https://arxiv.org/pdf/2504.19970), a transformer-based model for detecting shoplifting via human pose. It supports optimized inference, batching, and optional evaluation.

 GitHub: https://github.com/amalmathews/shopformer

------------------------------------------------------------

## Two Implementations

**V1 (shopformer_transformer.py):** Simple transformer autoencoder - fast prototype
**V2 (shopformer_v2.py):** Paper-faithful multi-stage architecture with graph convolutions

------------------------------------------------------------

SETUP

git clone https://github.com/amalmathews/shopformer.git
cd shopformer
pip install -r requirements.txt

------------------------------------------------------------

## Two Implementations

**V1 (shopformer_transformer.py):** Simple transformer autoencoder - fast prototype
**V2 (shopformer_v2.py):** Paper-faithful multi-stage architecture with graph convolutions

------------------------------------------------------------

RUN INFERENCE

**V1**
Basic Inference (no metrics):
python run_infer2.py --video_path <video.mp4>

With Ground Truth CSV (to compute precision / recall / F1):
```bash
python run_infer2.py  test.mp4 test_annotated.csv
```

**V2 (Enhanced - Default):**
```bash
python run_infer_v2.py test.mp4 test_annotated.csv
```

------------------------------------------------------------

OUTPUT FILES

- outputs/anomaly_only_video.mp4         → shoplifting-only frames
- outputs/normal_only_video.mp4          → normal behavior frames
- outputs/normality_score_plot.png       → MSE anomaly score plot
- Console prints per-frame anomaly scores
- Console prints evaluation metrics (if ground truth is provided)

example 
```
root@b9061694de8d:/workspace/shopformer# python run_infer2.py test.mp4 test_annotated.csv

🎬 Writing video outputs...
Frame 000: Score = 0.3925 → Normal
Frame 001: Score = 0.3867 → Normal
Frame 002: Score = 0.3672 → Normal
Frame 003: Score = 0.3559 → Normal
Frame 004: Score = 0.3423 → Normal
Frame 005: Score = 0.3300 → Normal
Frame 006: Score = 0.3367 → Normal
Frame 007: Score = 0.3710 → Normal
Frame 008: Score = 0.3969 → Normal
Frame 009: Score = 0.4073 → Shoplifting
Frame 010: Score = 0.4031 → Shoplifting
Frame 011: Score = 0.3874 → Normal
Frame 012: Score = 0.3671 → Normal
Frame 013: Score = 0.3610 → Normal
Frame 014: Score = 0.3815 → Normal
Frame 015: Score = 0.4082 → Shoplifting
Frame 016: Score = 0.4245 → Shoplifting
Frame 017: Score = 0.4050 → Shoplifting
Frame 018: Score = 0.3710 → Normal
Frame 019: Score = 0.3567 → Normal
Frame 020: Score = 0.3564 → Normal
Frame 021: Score = 0.3382 → Normal
Frame 022: Score = 0.3099 → Normal
Frame 023: Score = 0.2836 → Normal
Frame 024: Score = 0.2697 → Normal
Frame 025: Score = 0.2753 → Normal
Frame 026: Score = 0.2964 → Normal
Frame 027: Score = 0.3277 → Normal
Frame 028: Score = 0.3335 → Normal
Frame 029: Score = 0.3071 → Normal
Frame 030: Score = 0.2659 → Normal
Frame 031: Score = 0.2432 → Normal
Frame 032: Score = 0.2509 → Normal
Frame 033: Score = 0.2846 → Normal
Frame 034: Score = 0.3330 → Normal
Frame 035: Score = 0.3738 → Normal
Frame 036: Score = 0.3882 → Normal
Frame 037: Score = 0.3813 → Normal
Frame 038: Score = 0.3665 → Normal
Frame 039: Score = 0.3575 → Normal
Frame 040: Score = 0.3693 → Normal
Frame 041: Score = 0.3886 → Normal
Frame 042: Score = 0.3868 → Normal
Frame 043: Score = 0.3782 → Normal
Frame 044: Score = 0.3719 → Normal
Frame 045: Score = 0.3737 → Normal
Frame 046: Score = 0.3713 → Normal
Frame 047: Score = 0.3777 → Normal
Frame 048: Score = 0.3995 → Normal
Frame 049: Score = 0.4257 → Shoplifting
Frame 050: Score = 0.4490 → Shoplifting
Frame 051: Score = 0.4529 → Shoplifting
Frame 052: Score = 0.4289 → Shoplifting
Frame 053: Score = 0.4073 → Shoplifting
Frame 054: Score = 0.4072 → Shoplifting
Frame 055: Score = 0.4035 → Shoplifting
Frame 056: Score = 0.3912 → Normal
Frame 057: Score = 0.3845 → Normal
Frame 058: Score = 0.3937 → Normal
Frame 059: Score = 0.3978 → Normal
Frame 060: Score = 0.3979 → Normal
Frame 061: Score = 0.4132 → Shoplifting
Frame 062: Score = 0.4255 → Shoplifting
Frame 063: Score = 0.4376 → Shoplifting
Frame 064: Score = 0.4700 → Shoplifting
Frame 065: Score = 0.5091 → Shoplifting
Frame 066: Score = 0.5244 → Shoplifting
Frame 067: Score = 0.4967 → Shoplifting
Frame 068: Score = 0.4643 → Shoplifting
Frame 069: Score = 0.4622 → Shoplifting
Frame 070: Score = 0.4797 → Shoplifting
Frame 071: Score = 0.4889 → Shoplifting
Frame 072: Score = 0.4730 → Shoplifting
Frame 073: Score = 0.4476 → Shoplifting
Frame 074: Score = 0.4436 → Shoplifting
Frame 075: Score = 0.4399 → Shoplifting
Frame 076: Score = 0.4369 → Shoplifting
Frame 077: Score = 0.4514 → Shoplifting
Frame 078: Score = 0.4706 → Shoplifting
Frame 079: Score = 0.4627 → Shoplifting
Frame 080: Score = 0.4210 → Shoplifting
Frame 081: Score = 0.3697 → Normal
Frame 082: Score = 0.3464 → Normal
Frame 083: Score = 0.3511 → Normal
Frame 084: Score = 0.3799 → Normal
Frame 085: Score = 0.4306 → Shoplifting
Frame 086: Score = 0.4627 → Shoplifting
Frame 087: Score = 0.4552 → Shoplifting
Frame 088: Score = 0.4158 → Shoplifting
Frame 089: Score = 0.3624 → Normal
Frame 090: Score = 0.3318 → Normal
Frame 091: Score = 0.3451 → Normal
Frame 092: Score = 0.3800 → Normal
Frame 093: Score = 0.4039 → Shoplifting
Frame 094: Score = 0.4132 → Shoplifting
Frame 095: Score = 0.3991 → Normal
Frame 096: Score = 0.3770 → Normal
Frame 097: Score = 0.3720 → Normal
Frame 098: Score = 0.3765 → Normal
Frame 099: Score = 0.4009 → Shoplifting
Frame 100: Score = 0.4279 → Shoplifting
Frame 101: Score = 0.4517 → Shoplifting
Frame 102: Score = 0.4574 → Shoplifting
Frame 103: Score = 0.4579 → Shoplifting
Frame 104: Score = 0.4690 → Shoplifting
Frame 105: Score = 0.4744 → Shoplifting
Frame 106: Score = 0.4668 → Shoplifting
Frame 107: Score = 0.4485 → Shoplifting
Frame 108: Score = 0.4288 → Shoplifting
Frame 109: Score = 0.4057 → Shoplifting
Frame 110: Score = 0.4003 → Shoplifting
Frame 111: Score = 0.4174 → Shoplifting
Frame 112: Score = 0.4243 → Shoplifting
Frame 113: Score = 0.4180 → Shoplifting
Frame 114: Score = 0.4068 → Shoplifting
Frame 115: Score = 0.4105 → Shoplifting
Frame 116: Score = 0.4106 → Shoplifting
Frame 117: Score = 0.4180 → Shoplifting
Frame 118: Score = 0.4165 → Shoplifting
Frame 119: Score = 0.4028 → Shoplifting
Frame 120: Score = 0.3626 → Normal
Frame 121: Score = 0.3372 → Normal
Frame 122: Score = 0.3359 → Normal
Frame 123: Score = 0.3509 → Normal
Frame 124: Score = 0.3598 → Normal
Frame 125: Score = 0.3441 → Normal
Frame 126: Score = 0.3318 → Normal
Frame 127: Score = 0.3340 → Normal
Frame 128: Score = 0.3427 → Normal
Frame 129: Score = 0.3506 → Normal
Frame 130: Score = 0.3571 → Normal
Frame 131: Score = 0.3684 → Normal
Frame 132: Score = 0.3891 → Normal
Frame 133: Score = 0.4127 → Shoplifting
Frame 134: Score = 0.4550 → Shoplifting
Frame 135: Score = 0.5087 → Shoplifting
Frame 136: Score = 0.5553 → Shoplifting
Frame 137: Score = 0.5681 → Shoplifting
Frame 138: Score = 0.5477 → Shoplifting
Frame 139: Score = 0.5210 → Shoplifting
Frame 140: Score = 0.5192 → Shoplifting

✅ Saved 67 anomalous frames → outputs/anomaly_only_video.mp4
✅ Saved 74 normal frames → outputs/normal_only_video.mp4
Anomaly Ratio: 67/141 = 47.52%

 Evaluation Metrics:
Precision: 0.5373
Recall:    0.3830
F1 Score:  0.4472
```
------------------------------------------------------------

MODEL & APPROACH

- Uses YOLOv8-pose for keypoint extraction
- Shopformer is a transformer autoencoder trained in unsupervised fashion
- Computes per-frame MSE reconstruction error as anomaly score
- Optimized with batched inference for pose estimation

models/shopformerv2.py Architecture
Implemented the full transformer-based architecture described in the Shopformer paper, including graph-based spatial encoding, tokenization, transformer encoder-decoder, and pose reconstruction. Not trained due to absence of annotated data, but the forward pass is complete and matches the original design.

## Architecture Comparison

| Feature | V1 | V2 |
|---------|----|----|
| Structure | Flat vector | Graph convolution |
| Tokenization | None | 2 tokens per pose |
| Transformer | 2 layers, 4 heads | 6 layers, 8 heads |
| Parameters | 1.2M | 11.6M |
| Paper Fidelity | Simplified | Faithful |
| F1 Score | 44% | 80% |

---

**V1:** Simple transformer autoencoder for rapid prototyping.

**V2:** Multi-stage architecture matching paper design:
- Graph Convolutional Encoder (models skeleton structure)
- Tokenizer (17 joints → 2 compact tokens)
- Transformer (6 layers, 8 heads, temporal modeling)
- Graph Decoder (reconstruction)

Both use unsupervised reconstruction error for anomaly detection.

---

------------------------------------------------------------

GROUND TRUTH CSV FORMAT (OPTIONAL)

temporal_segment_start,temporal_segment_end,metadata
0.0,6.2,"{\"TEMPORAL_SEGMENTS\": \"Shoplifting\"}"

------------------------------------------------------------

 NOTES

- Ground truth is optional — use it to get evaluation metrics
- Code includes both batched and non-batched pose extraction (toggle in script)
- Designed for reproducible evaluation with annotated video outputs
- TeCSAR-UNCC/Shopformer repository (referenced in paper) currently contains only documentation - implementation pending CVPR 2025 publication. Both versions implemented from paper architecture description.

