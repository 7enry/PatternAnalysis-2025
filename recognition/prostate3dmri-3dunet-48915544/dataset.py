"""
HipMRI 3D Dataset Loader for Prostate MRI Segmentation
======================================================

This module handles loading, preprocessing, and data augmentation of the HipMRI 3D prostate MRI dataset.
It provides PyTorch Dataset classes for training, validation, and testing with 3D volumes.

Author: Henry
Course: COMP3710 Pattern Analysis
"""
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom, rotate
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict, Any
import matplotlib.pyplot as plt
import warnings
import random
warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# HipMRI dataset class definitions
HIPMRI_CLASSES = {
    0: 'Background',
    1: 'Body Outline', 
    2: 'Bone',
    3: 'Bladder',
    4: 'Rectum',
    5: 'Prostate'
}
NUM_CLASSES = len(HIPMRI_CLASSES)
DEFAULT_TARGET_SHAPE = (96, 96, 96)

# File naming patterns
MRI_SUFFIX = '_LFOV.nii.gz'
LABEL_SUFFIX = '_SEMANTIC.nii.gz'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def resize_image(image: np.ndarray, target_shape: Tuple[int, int, int], order: int = 1) -> np.ndarray:
    """
    Resize 3D volume to target dimensions using scipy zoom.
    
    Args:
        image: 3D volume (D, H, W)
        target_shape: Target (D, H, W) 
        order: 0=nearest (masks), 1=linear (images)
    Returns:
        Resized volume
    """
    if image.shape == target_shape:
        return image
    
    # Calculate per-axis scale factors
    scale_factors = [n / o for n, o in zip(target_shape, image.shape)]
    return zoom(image, scale_factors, order=order, mode='constant')


def load_nifti_volume(file_path: str) -> np.ndarray:
    """
    Load NIfTI volume and handle common dimension issues.
    
    Args:
        file_path: Path to NIfTI file
    Returns:
        3D volume as float32 array
    """
    nifti_image = nib.load(file_path)
    volume = nifti_image.get_fdata(caching='unchanged')
    
    # Remove 4th dimension if present (common in HipMRI data)
    if len(volume.shape) == 4:
        volume = volume[:, :, :, 0]
    
    return volume.astype(np.float32)


def mask_to_onehot(mask: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """
    Convert integer segmentation mask to one-hot encoding.
    
    Args:
        mask: Integer labels (D, H, W)
        num_classes: Number of classes
    Returns:
        One-hot mask (num_classes, D, H, W)
    """
    onehot = np.zeros((num_classes, *mask.shape), dtype=np.float32)
    mask_int = mask.astype(np.int32)
    
    for i in range(num_classes):
        onehot[i] = (mask_int == i).astype(np.float32)
    
    return onehot


def normalize_volume(volume: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize 3D volume using specified method.
    
    Args:
        volume: 3D volume (D, H, W)
        method: 'zscore' or 'minmax'
    Returns:
        Normalized volume
    """
    if method == 'zscore':
        return (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
    elif method == 'minmax':
        return (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class Augment:
    """
    3D data augmentation for prostate MRI using scipy (no external deps).
    Applies flips, rotations, and scaling with synchronized transforms.
    """
    def __init__(self, target_shape=(96, 96, 96)):
        self.target_shape = target_shape

    def apply_augmentation(self, image: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """
        Apply random 3D augmentations with proper interpolation.
        
        Args:
            image: 3D volume (D, H, W)
            is_mask: Use nearest-neighbor for masks, linear for images
        Returns:
            Augmented volume
        """
        # Set interpolation: nearest for masks, linear for images
        interp_order = 0 if is_mask else 1
        
        # Random flips: 50% chance per axis
        if random.random() > 0.5:
            image = np.flip(image, axis=0)  # Flip depth
        if random.random() > 0.5:
            image = np.flip(image, axis=1)  # Flip height
        if random.random() > 0.5:
            image = np.flip(image, axis=2)  # Flip width
        
        # Random rotation: ±5 degrees around random axis
        if random.random() > 0.5:
            axis = random.choice([0, 1, 2])
            angle = random.uniform(-5, 5)
            image = rotate(image, angle, axes=(axis, (axis + 1) % 3), 
                          reshape=False, order=interp_order, mode='constant')
        
        # Random scaling: 90-110% with crop/pad to maintain size
        if random.random() > 0.5:
            scale_factor = random.uniform(0.9, 1.1)
            current_shape = image.shape
            new_shape = tuple(int(s * scale_factor) for s in current_shape)
            
            # Resize using zoom
            zoom_factors = [n / o for n, o in zip(new_shape, current_shape)]
            image = zoom(image, zoom_factors, order=interp_order, mode='constant')
            
            # Restore original size
            if scale_factor > 1.0:
                # Crop from center
                slices = tuple(slice(0, s) for s in self.target_shape)
                image = image[slices]
            else:
                # Pad with zeros
                pad_width = [(0, max(0, target - current)) for target, current in zip(self.target_shape, image.shape)]
                image = np.pad(image, pad_width, mode='constant')
                # Crop to exact size
                slices = tuple(slice(0, s) for s in self.target_shape)
                image = image[slices]
        
        return image