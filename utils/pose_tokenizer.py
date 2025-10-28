import numpy as np
import torch

def normalize_keypoints(keypoints, image_shape):
    H, W = image_shape
    keypoints = np.array(keypoints)
    keypoints[..., 0] /= W
    keypoints[..., 1] /= H
    keypoints = keypoints.reshape(len(keypoints), -1)
    return torch.tensor(keypoints, dtype=torch.float32)
