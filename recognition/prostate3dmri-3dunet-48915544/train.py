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


def main():
    parser = argparse.ArgumentParser(description="COMP3710 Prostate 3D MRI Segmentation Training")
    parser.add_argument('--data_dir', type=str, default='/home/groups/comp3710/HipMRI_Study_open',
                        help='Base directory of the HipMRI dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Input batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Number of batches to accumulate gradients over')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints and logs')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to a checkpoint to resume training from')
    args = parser.parse_args()

    # Setup device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning: CUDA not found. Using CPU.")
    print(f"Using device: {DEVICE}")

    # Setup checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    results_dir = os.path.join(args.checkpoint_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, 'tensorboard'))

    print("\n Training Configuration")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Gradient accumulation steps: {args.accumulation_steps}")
    print(f"   Number of epochs: {args.num_epochs}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        base_dir=args.data_dir,
        target_shape=(96, 96, 96),
        batch_size=args.batch_size,
        num_workers=4,
        augment_train=True
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Initialize the 3D U-Net model
    model = UNet3D(
        in_channels=1,
        out_channels=NUM_CLASSES,
        init_features=32
    ).to(DEVICE)

    # Select the Dice loss function for training
    criterion = dice_loss

    # Setup the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler()  # For mixed-precision training

    # Load checkpoint if resuming
    start_epoch = 0
    best_val_dice = 0.0
    train_losses = []
    val_losses = []
    avg_val_dice_scores = []
    class_val_dice_scores = {c: [] for c in range(NUM_CLASSES)}

    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_dice = checkpoint['best_val_dice']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        avg_val_dice_scores = checkpoint['avg_val_dice_scores']
        class_val_dice_scores = checkpoint['class_val_dice_scores']
        print(f"Resumed from epoch {start_epoch}, best validation Dice: {best_val_dice:.4f}")

    # Training loop
    print("\n Starting Training")
    for epoch in range(start_epoch, args.num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        
        for i, (mri_data, label_data) in enumerate(train_pbar):
            mri_data = mri_data.to(DEVICE)  # (B, 1, D, H, W)
            label_data = label_data.to(DEVICE)  # (B, 6, D, H, W)

            with autocast():
                outputs = model(mri_data)  # Forward pass (B, 6, D, H, W)
                loss = criterion(outputs, label_data, NUM_CLASSES) / args.accumulation_steps

            scaler.scale(loss).backward()  # Backpropagate errors with scaling

            if (i + 1) % args.accumulation_steps == 0:
                scaler.step(optimizer)  # Update model parameters
                scaler.update()
                optimizer.zero_grad()  # Reset gradients for next accumulation

            train_loss += loss.item() * args.accumulation_steps
            train_pbar.set_postfix({'loss': f'{loss.item() * args.accumulation_steps:.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)

        torch.cuda.empty_cache()  # Clear unused memory

        # Validation phase
        model.eval()
        val_loss = 0
        total_val_dice_per_class = {c: 0 for c in range(NUM_CLASSES)}
        num_val_batches = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")
            for mri_data, label_data in val_pbar:
                mri_data = mri_data.to(DEVICE)
                label_data = label_data.to(DEVICE)

                with autocast():
                    outputs = model(mri_data)
                    loss = criterion(outputs, label_data, NUM_CLASSES)
                    val_loss += loss.item()

                    # Calculate Dice score per class
                    dice_per_class = calculate_dice_per_class(outputs, label_data, NUM_CLASSES)
                    for c in range(NUM_CLASSES):
                        total_val_dice_per_class[c] += dice_per_class[c]

                num_val_batches += 1
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        # Calculate average Dice scores
        avg_dice_per_class = {c: total_val_dice_per_class[c] / num_val_batches for c in range(NUM_CLASSES)}
        avg_dice_score = np.mean([avg_dice_per_class[c] for c in range(NUM_CLASSES)])
        avg_val_dice_scores.append(avg_dice_score)
        writer.add_scalar('Dice/Average', avg_dice_score, epoch)

        # Store per-class Dice scores
        for c in range(NUM_CLASSES):
            class_val_dice_scores[c].append(avg_dice_per_class[c])
            writer.add_scalar(f'Dice/{CLASS_NAMES[c]}', avg_dice_per_class[c], epoch)

        print(f"\nEpoch [{epoch + 1}/{args.num_epochs}]")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  Average Validation Dice: {avg_dice_score:.4f}")
        for c in range(NUM_CLASSES):
            print(f"  {CLASS_NAMES[c]} Dice: {avg_dice_per_class[c]:.4f}")

        # Save best model
        is_best = avg_dice_score > best_val_dice
        if is_best:
            best_val_dice = avg_dice_score

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_dice': best_val_dice,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'avg_val_dice_scores': avg_val_dice_scores,
            'class_val_dice_scores': class_val_dice_scores,
        }, is_best, args.checkpoint_dir, epoch)

        if is_best:
            print(f"   New best model saved -> Val Dice: {best_val_dice:.4f}")

        # Plot metrics every 10 epochs
        if (epoch + 1) % 10 == 0:
            # 1. Training vs Validation Loss
            plot_metrics(
                {'Training Loss': train_losses, 'Validation Loss': val_losses},
                'Training and Validation Loss',
                'Loss',
                'loss_train_val.png',
                results_dir
            )
            
            # 2. Average Validation Dice
            plot_metrics(
                {'Average Dice': avg_val_dice_scores},
                'Average Validation Dice Score',
                'Dice Score',
                'dice_val_mean.png',
                results_dir
            )
            
            # 3. Per-class Validation Dice
            plot_metrics(
                {CLASS_NAMES[c]: class_val_dice_scores[c] for c in range(NUM_CLASSES)},
                'Per-Class Validation Dice Scores',
                'Dice Score',
                'dice_val_per_class.png',
                results_dir
            )

    # Final visualization and test evaluation
    
    # 1. Training vs Validation Loss (final)
    plot_metrics(
        {'Training Loss': train_losses, 'Validation Loss': val_losses},
        'Training and Validation Loss',
        'Loss',
        'loss_train_val.png',
        results_dir
    )
    print(f"    loss_train_val.png")
    
    # 2. Average Validation Dice (final)
    plot_metrics(
        {'Average Dice': avg_val_dice_scores},
        'Average Validation Dice Score',
        'Dice Score',
        'dice_val_mean.png',
        results_dir
    )
    print(f"    dice_val_mean.png")
    
    # 3. Per-class Validation Dice (final)
    plot_metrics(
        {CLASS_NAMES[c]: class_val_dice_scores[c] for c in range(NUM_CLASSES)},
        'Per-Class Validation Dice Scores',
        'Dice Score',
        'dice_val_per_class.png',
        results_dir
    )
    print(f"    dice_val_per_class.png")
    
    # 4. Qualitative Examples
    plot_qualitative_examples(model, val_loader, DEVICE, results_dir, num_examples=3)
    print(f"    qualitative_examples_case*.png")
    
    # 5. Test Set Evaluation
    model.eval()
    test_dice_scores = {c: [] for c in range(NUM_CLASSES)}
    
    with torch.no_grad():
        for mri_data, label_data in test_loader:
            mri_data = mri_data.to(DEVICE)
            label_data = label_data.to(DEVICE)
            
            outputs = model(mri_data)
            dice_per_class = calculate_dice_per_class(outputs, label_data, NUM_CLASSES)
            
            for c in range(NUM_CLASSES):
                test_dice_scores[c].append(dice_per_class[c])
    
    # Calculate mean test Dice scores
    mean_test_dice = {c: np.mean(test_dice_scores[c]) for c in range(NUM_CLASSES)}
    overall_test_dice = np.mean(list(mean_test_dice.values()))
    
    print(f"    Test Set Results:")
    print(f"      Overall Dice: {overall_test_dice:.4f}")
    for c in range(NUM_CLASSES):
        print(f"      {CLASS_NAMES[c]}: {mean_test_dice[c]:.4f}")
    
    # 6. Test Dice Boxplot
    plot_dice_boxplot(test_dice_scores, results_dir)
    print(f"    test_dice_summary.png")

    # Save the final trained model (weights-only) under results/
    final_dir = os.path.join(args.checkpoint_dir, 'results')
    os.makedirs(final_dir, exist_ok=True)
    final_model_path = os.path.join(final_dir, 'final.pth')
    torch.save(model.state_dict(), final_model_path)
        
    print(f"\n Training Successfully Completed")
    print(f" Best validation Dice: {best_val_dice:.4f}")
    print(f" Test set Dice: {overall_test_dice:.4f}")
    print(f" Final model saved: {final_model_path}")
    print(f" Best model saved: {os.path.join(args.checkpoint_dir, 'best.pth')}")
    print(f" Plot directory: {results_dir}")
    writer.close()


if __name__ == "__main__":
    main()