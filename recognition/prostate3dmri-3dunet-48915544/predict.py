"""
Prediction and Evaluation Script for Prostate 3D MRI Segmentation
================================================================

This script loads a trained 3D U-Net model and evaluates it on the test set,
computing Dice scores and generating visualizations.

Author: Henry
Course: COMP3710 Pattern Analysis
"""

import os
import argparse
import numpy as np
import torch
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from modules import UNet3D, dice_coefficient
from dataset import create_data_loaders

# ============================================================================
# CONSTANTS
# ============================================================================

CLASS_NAMES = ['Background', 'Body Outline', 'Bone', 'Bladder', 'Rectum', 'Prostate']
NUM_CLASSES = len(CLASS_NAMES)
COLORS = ['black', 'blue', 'green', 'yellow', 'red', 'purple']


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Target device (cuda/cpu)
    Returns:
        Loaded UNet3D model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create model architecture
    model = UNet3D(in_channels=1, out_channels=NUM_CLASSES)
    
    # Load weights (handle both full checkpoint and state_dict-only)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    return model


def compute_dice_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_CLASSES, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Dice coefficient for each class.
    
    Args:
        pred: Predicted labels (B, D, H, W)
        target: Ground truth labels (B, D, H, W)
        num_classes: Number of classes
        smooth: Smoothing factor
    Returns:
        Dice scores per class as tensor
    """
    dice_scores = torch.zeros(num_classes, device=pred.device)
    
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores[c] = dice
    
    return dice_scores


def visualize_volume_slices(img: np.ndarray, pred: np.ndarray, title_img: str, title_pred: str, save_path: str):
    """
    Visualize 3D volume as maximum intensity projection.
    
    Args:
        img: Input MRI volume (D, H, W)
        pred: Predicted segmentation (D, H, W)
        title_img: Title for input image
        title_pred: Title for prediction
        save_path: Path to save visualization
    """
    # Maximum intensity projection along depth axis
    img_proj = np.max(img, axis=0)
    pred_proj = np.max(pred, axis=0)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    axs[0].imshow(img_proj, cmap='gray')
    axs[0].set_title(title_img)
    axs[0].axis('off')
    
    cmap = ListedColormap(COLORS[:NUM_CLASSES])
    axs[1].imshow(pred_proj, cmap=cmap, interpolation='nearest', vmin=0, vmax=NUM_CLASSES-1)
    axs[1].set_title(title_pred)
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model: torch.nn.Module, test_loader, device: torch.device) -> dict:
    """
    Evaluate model on test set and compute metrics.
    
    Args:
        model: Trained UNet3D model
        test_loader: Test data loader
        device: Computation device
    Returns:
        Dictionary with evaluation results and best/worst cases
    """
    model.eval()
    
    # Track metrics
    total_dice_per_class = {c: 0.0 for c in range(NUM_CLASSES)}
    best_images = {c: (None, -1.0) for c in range(NUM_CLASSES)}
    worst_images = {c: (None, 1.0) for c in range(NUM_CLASSES)}
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (mri_data, label_data) in enumerate(test_loader):
            mri_data = mri_data.to(device)
            label_data = label_data.to(device)
            
            with autocast():
                outputs = model(mri_data)
                preds = torch.argmax(outputs, dim=1)  # (B, D, H, W)
            
            # Convert one-hot labels to class indices
            target_labels = torch.argmax(label_data, dim=1)  # (B, D, H, W)
            
            # Compute Dice per class for this batch
            for b in range(mri_data.size(0)):
                batch_pred = preds[b]
                batch_target = target_labels[b]
                
                dice_scores = compute_dice_per_class(batch_pred.unsqueeze(0), batch_target.unsqueeze(0))
                
                # Update totals
                for c in range(NUM_CLASSES):
                    total_dice_per_class[c] += dice_scores[c].item()
                    
                    # Track best/worst per class
                    score = dice_scores[c].item()
                    if score > best_images[c][1]:
                        best_images[c] = (mri_data[b].cpu().numpy(), score)
                    if score < worst_images[c][1]:
                        worst_images[c] = (mri_data[b].cpu().numpy(), score)
            
            num_batches += mri_data.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Calculate averages
    avg_dice_per_class = {c: total_dice_per_class[c] / num_batches for c in range(NUM_CLASSES)}
    avg_dice_overall = np.mean([avg_dice_per_class[c] for c in range(NUM_CLASSES)])
    
    return {
        'avg_dice_per_class': avg_dice_per_class,
        'avg_dice_overall': avg_dice_overall,
        'best_images': best_images,
        'worst_images': worst_images,
        'num_samples': num_batches
    }


def generate_visualizations(model: torch.nn.Module, results: dict, output_dir: str, device: torch.device):
    """
    Generate visualization plots for best/worst cases per class.
    
    Args:
        model: Trained model
        results: Evaluation results dictionary
        output_dir: Directory to save visualizations
        device: Computation device
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    # Visualize best/worst per class
    for c in range(NUM_CLASSES):
        best_img, best_score = results['best_images'][c]
        worst_img, worst_score = results['worst_images'][c]
        
        if best_img is not None:
            # Get prediction for best case
            with torch.no_grad():
                img_tensor = torch.from_numpy(best_img).unsqueeze(0).to(device)
                with autocast():
                    outputs = model(img_tensor)
                    best_pred = torch.argmax(outputs, dim=1)[0].cpu().numpy()
            
            visualize_volume_slices(
                best_img[0], best_pred,
                f'Best {CLASS_NAMES[c]} (Dice={best_score:.4f})',
                f'Prediction {CLASS_NAMES[c]} (Dice={best_score:.4f})',
                os.path.join(output_dir, f'best_class_{c}.png')
            )
        
        if worst_img is not None:
            # Get prediction for worst case
            with torch.no_grad():
                img_tensor = torch.from_numpy(worst_img).unsqueeze(0).to(device)
                with autocast():
                    outputs = model(img_tensor)
                    worst_pred = torch.argmax(outputs, dim=1)[0].cpu().numpy()
            
            visualize_volume_slices(
                worst_img[0], worst_pred,
                f'Worst {CLASS_NAMES[c]} (Dice={worst_score:.4f})',
                f'Prediction {CLASS_NAMES[c]} (Dice={worst_score:.4f})',
                os.path.join(output_dir, f'worst_class_{c}.png')
            )
    
    print(f" Visualizations saved to: {output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Prostate 3D MRI Segmentation Model')
    parser.add_argument('--data_dir', type=str, default='/home/groups/comp3710/HipMRI_Study_open',
                       help='Path to HipMRI dataset')
    parser.add_argument('--checkpoint_path', type=str, 
                       default='/home/Student/s4891554/checkpoints/results/final.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                       help='Directory to save results and visualizations')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model

    model = load_model(args.checkpoint_path, device)
    
    # Create test data loader
    _, _, test_loader = create_data_loaders(
        base_dir=args.data_dir,
        target_shape=(96, 96, 96),
        batch_size=args.batch_size,
        num_workers=4,
        augment_train=False
    )
    
    print(f"   Test batches: {len(test_loader)}")
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device)
    
    # Print results
    print(f"\n Evaluation Results:")
    print("=" * 50)
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]}: Average Dice = {results['avg_dice_per_class'][c]:.4f}")
    print(f"\n  Overall Average Dice: {results['avg_dice_overall']:.4f}")
    print(f"  Total samples evaluated: {results['num_samples']}")
    
    # Generate visualizations
    generate_visualizations(model, results, args.output_dir, device)
    
    print(f"\n Evaluation Completed")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
