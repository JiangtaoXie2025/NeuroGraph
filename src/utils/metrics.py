# utils/metrics.py
#
# Contains evaluation metrics. Examples include MPJPE, PCK, or
# domain-specific metrics for brain imaging tasks.

import torch
import math

def mpjpe(predictions, targets):
    """
    Mean Per Joint Position Error (MPJPE)
    predictions: [batch_size, num_joints, 3]
    targets: [batch_size, num_joints, 3]
    returns: scalar MPJPE value
    """
    return torch.mean(torch.norm(predictions - targets, dim=-1))

def pck(predictions, targets, threshold=150.0):
    """
    Percentage of Correct Keypoints (PCK).
    threshold can be adapted for scale.
    predictions: [batch_size, num_joints, 2] (for 2D) or [batch_size, num_joints, 3] (3D)
    targets: same shape as predictions
    returns: fraction of keypoints under distance threshold
    """
    distances = torch.norm(predictions - targets, dim=-1)
    correct = (distances < threshold).float()
    return torch.mean(correct)

def custom_metric(output, target):
    """
    Placeholder for a domain-specific metric, such as correlation
    with EEG signals or classification accuracy for neurological states.
    """
    # Example: classification accuracy
    preds = torch.argmax(output, dim=1)
    correct = (preds == target).sum().item()
    return correct / target.size(0)
