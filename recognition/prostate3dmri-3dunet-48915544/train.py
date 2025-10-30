"""
3D U-Net Training Script for Prostate 3D MRI Segmentation
===========================================================

This script trains a 3D U-Net model on the HipMRI 3D prostate MRI dataset for volumetric segmentation.
It includes proper data loading, training loop, validation, and visualization.

Author: Henry
Course: COMP3710 Pattern Analysis
"""

import argparse
import os
import time
import json
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import create_data_loaders
from modules import UNet3D, dice_coefficient, dice_loss

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define class names for visualization and reporting
# Classes: Background, Body Outline, Bone, Bladder, Rectum, Prostate
CLASS_NAMES = ['Background', 'Body Outline', 'Bone', 'Bladder', 'Rectum', 'Prostate']
NUM_CLASSES = len(CLASS_NAMES)


def calculate_dice_per_class(pred, target, num_classes=6, smooth=1e-6):
    """
    Calculate Dice coefficient for each class separately.
        
        Args:
        pred: Predicted logits (B, C, D, H, W)
        target: Target one-hot masks (B, C, D, H, W)
        num_classes: Number of classes
        smooth: Smoothing factor
            
        Returns:
        Dictionary of Dice scores per class
    """
    pred_probs = torch.softmax(pred, dim=1)
    pred_labels = torch.argmax(pred_probs, dim=1)  # (B, D, H, W)
    target_labels = torch.argmax(target, dim=1)  # (B, D, H, W)
    
    dice_scores = {}
    
    for c in range(num_classes):
        pred_c = (pred_labels == c).float()
        target_c = (target_labels == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores[c] = dice_score.item()
    
    return dice_scores


if __name__ == "__main__":
    main()