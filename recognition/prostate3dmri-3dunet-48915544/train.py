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


def save_checkpoint(state, is_best, checkpoint_dir, epoch):
    """Save minimal checkpoint: only best weights; no per-epoch files."""
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best.pth')
        # Save weights only to minimize disk usage
        torch.save(state['model_state_dict'], best_filepath)


def plot_metrics(metrics, title, ylabel, filename, results_dir):
    """Plots training metrics."""
    plt.figure(figsize=(10, 6))
    for label, values in metrics.items():
        plt.plot(values, label=label)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_qualitative_examples(model, val_loader, device, results_dir, num_examples=3):
    """Plot qualitative examples: input, GT, prediction."""
    model.eval()
    with torch.no_grad():
        for i, (mri_data, label_data) in enumerate(val_loader):
            if i >= num_examples:
                break
                
            mri_data = mri_data.to(device)
            label_data = label_data.to(device)
            
            # Get prediction
            outputs = model(mri_data)
            pred_labels = torch.argmax(outputs, dim=1)  # (B, D, H, W)
            gt_labels = torch.argmax(label_data, dim=1)  # (B, D, H, W)
            
            # Take middle slice for visualization
            batch_idx = 0
            slice_idx = mri_data.shape[2] // 2
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Input MRI
            axes[0].imshow(mri_data[batch_idx, 0, slice_idx].cpu().numpy(), cmap='gray')
            axes[0].set_title('Input MRI')
            axes[0].axis('off')
            
            # Ground Truth
            im1 = axes[1].imshow(gt_labels[batch_idx, slice_idx].cpu().numpy(), cmap='tab10', vmin=0, vmax=5)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            im2 = axes[2].imshow(pred_labels[batch_idx, slice_idx].cpu().numpy(), cmap='tab10', vmin=0, vmax=5)
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'qualitative_examples_case{i:02d}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()


def plot_dice_boxplot(test_dice_scores, results_dir):
    """Plot Dice score boxplot across test set."""
    plt.figure(figsize=(12, 8))
    
    # Prepare data for boxplot
    data = []
    labels = []
    for class_idx, scores in test_dice_scores.items():
        data.append(scores)
        labels.append(CLASS_NAMES[class_idx])
    
    plt.boxplot(data, labels=labels)
    plt.title('Test Set Dice Scores by Class')
    plt.ylabel('Dice Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'test_dice_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()