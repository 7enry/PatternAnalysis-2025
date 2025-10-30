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


# ============================================================================
# FILE MATCHING AND DATASET UTILITIES
# ============================================================================

def match_mri_label_pairs(mri_dir: str, labels_dir: str) -> List[Dict[str, str]]:
    """
    Match MRI and label files by patient_week pattern.
    
    Args:
        mri_dir: Directory with *_LFOV.nii.gz files
        labels_dir: Directory with *_SEMANTIC.nii.gz files
    Returns:
        List of matched file pairs
    """
    mri_files = sorted([f for f in os.listdir(mri_dir) if f.endswith(MRI_SUFFIX)])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(LABEL_SUFFIX)])
    
    volume_pairs = []
    for mri_file in mri_files:
        # Extract: B006_Week0_LFOV.nii.gz -> B006_Week0
        base_name = mri_file.replace(MRI_SUFFIX, '')
        expected_label = f"{base_name}{LABEL_SUFFIX}"
        
        if expected_label in label_files:
            volume_pairs.append({
                'mri_path': os.path.join(mri_dir, mri_file),
                'label_path': os.path.join(labels_dir, expected_label),
                'patient_week': base_name
            })
        else:
            print(f"Warning: No matching label found for {mri_file}")
    
    return volume_pairs


def create_dataset_splits(volume_pairs: List[Dict[str, str]], 
                         train_split: float = 0.7, 
                         val_split: float = 0.15) -> Tuple[List[int], List[int], List[int]]:
    """
    Create random train/val/test splits from volume pairs.
    
    Args:
        volume_pairs: List of matched file pairs
        train_split: Training fraction
        val_split: Validation fraction
    Returns:
        (train_indices, val_indices, test_indices)
    """
    total_size = len(volume_pairs)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices


# ============================================================================
# PYTORCH DATASET CLASS
# ============================================================================

class HipMRI3DDataset(Dataset):
    """
    PyTorch Dataset for HipMRI 3D prostate MRI segmentation.
    
    Structure: semantic_MRs/*_LFOV.nii.gz + semantic_labels_only/*_SEMANTIC.nii.gz
    Matches files by patient_week pattern (e.g., B006_Week0).
    """
    def __init__(self, mri_dir: str, labels_dir: str, target_shape: Tuple[int, int, int] = DEFAULT_TARGET_SHAPE, 
                 augment: bool = True, normalize: bool = True, normalize_method: str = 'zscore'):
        """
        Initialize HipMRI dataset with modular file matching.
        
        Args:
            mri_dir: Directory with *_LFOV.nii.gz files
            labels_dir: Directory with *_SEMANTIC.nii.gz files  
            target_shape: Resize target (D, H, W)
            augment: Apply random transforms
            normalize: Apply normalization
            normalize_method: 'zscore' or 'minmax'
        """
        # Use modular file matching
        self.volume_pairs = match_mri_label_pairs(mri_dir, labels_dir)
        
        self.mri_dir = mri_dir
        self.labels_dir = labels_dir
        self.target_shape = target_shape
        self.augment = augment
        self.normalize = normalize
        self.normalize_method = normalize_method
        
        print(f"Loaded {len(self.volume_pairs)} 3D volume pairs from NIfTI files")

    def __len__(self) -> int:
        return len(self.volume_pairs)

    def _load_nifti_volume(self, file_path: str) -> np.ndarray:
        """Load NIfTI volume using utility function."""
        return load_nifti_volume(file_path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get MRI volume and segmentation mask with preprocessing.
        
        Args:
            idx: Sample index
        Returns:
            (MRI, Mask): (1,D,H,W) normalized MRI, (6,D,H,W) one-hot mask
        """
        volume_info = self.volume_pairs[idx]
        
        # Load NIfTI volumes
        mri_data = self._load_nifti_volume(volume_info['mri_path'])  # (D, H, W)
        label_data = self._load_nifti_volume(volume_info['label_path'])  # (D, H, W)
        
        # Resize: linear for MRI, nearest for masks
        mri_data = resize_image(mri_data, self.target_shape, order=1)
        label_data = resize_image(label_data, self.target_shape, order=0)
        
        # Clip labels to valid range (0-5 for 6 classes)
        label_data = np.clip(label_data, 0, 5).astype(np.uint8)
        
        # Synchronized augmentation: same seed for image and mask
        if self.augment:
            seed = random.randint(0, 2**32 - 1)
            augmenter = Augment(target_shape=self.target_shape)
            
            random.seed(seed)
            mri_data = augmenter.apply_augmentation(mri_data, is_mask=False)
            
            random.seed(seed)
            label_data = augmenter.apply_augmentation(label_data, is_mask=True)
        
        # Normalize MRI data
        if self.normalize:
            mri_data = normalize_volume(mri_data, method=self.normalize_method)
        
        # Convert to one-hot encoding
        mask_onehot = mask_to_onehot(label_data, num_classes=NUM_CLASSES)
        
        # Convert to PyTorch tensors
        mri_tensor = torch.tensor(mri_data, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
        mask_tensor = torch.tensor(mask_onehot, dtype=torch.float32)  # (6, D, H, W)
        
        return mri_tensor, mask_tensor


# ============================================================================
# DATALOADER CREATION AND DATASET MANAGEMENT
# ============================================================================

def create_data_loaders(
    base_dir: str,
    target_shape: Tuple[int, int, int] = DEFAULT_TARGET_SHAPE,
    batch_size: int = 4,
    num_workers: int = 4,
    augment_train: bool = True,
    train_split: float = 0.7,
    val_split: float = 0.15,
    normalize_method: str = 'zscore'
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Create train/val/test DataLoaders with modular split creation.
    
    Args:
        base_dir: HipMRI root (semantic_MRs/ + semantic_labels_only/)
        target_shape: Resize target (D, H, W)
        batch_size: Batch size for training
        num_workers: Worker processes
        augment_train: Augment training data only
        train_split: Training fraction (0.7)
        val_split: Validation fraction (0.15)
        normalize_method: Normalization method
    Returns:
        (train_loader, val_loader, test_loader)
    """
    mri_dir = os.path.join(base_dir, 'semantic_MRs')
    seg_dir = os.path.join(base_dir, 'semantic_labels_only')
    
    # Check directories exist
    if not os.path.exists(mri_dir):
        print(f"Warning: {mri_dir} not found")
        return None, None, None
    if not os.path.exists(seg_dir):
        print(f"Warning: {seg_dir} not found")
        return None, None, None
    
    # Get all volume pairs using modular matching
    volume_pairs = match_mri_label_pairs(mri_dir, seg_dir)
    
    # Create splits using modular function
    train_indices, val_indices, test_indices = create_dataset_splits(
        volume_pairs, train_split, val_split
    )
    
    # Create datasets with appropriate settings
    train_dataset = HipMRI3DDataset(mri_dir, seg_dir, target_shape, 
                                   augment=augment_train, normalize=True, 
                                   normalize_method=normalize_method)
    train_dataset.volume_pairs = [volume_pairs[i] for i in train_indices]
    
    val_dataset = HipMRI3DDataset(mri_dir, seg_dir, target_shape, 
                                 augment=False, normalize=True, 
                                 normalize_method=normalize_method)
    val_dataset.volume_pairs = [volume_pairs[i] for i in val_indices]
    
    test_dataset = HipMRI3DDataset(mri_dir, seg_dir, target_shape, 
                                  augment=False, normalize=True, 
                                  normalize_method=normalize_method)
    test_dataset.volume_pairs = [volume_pairs[i] for i in test_indices]
    
    print(f" Dataset splits:")
    print(f"   Train: {len(train_dataset)} volumes")
    print(f"   Val: {len(val_dataset)} volumes")
    print(f"   Test: {len(test_dataset)} volumes")
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# DATASET INFORMATION AND UTILITIES
# ============================================================================

def print_dataset_info(base_dir: str) -> None:
    """
    Print HipMRI dataset statistics using modular functions.
    
    Args:
        base_dir: HipMRI root directory
    """
    print("HipMRI 3D Dataset Information:")
    print("=" * 50)
    
    # Error check
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return
    
    mri_dir = os.path.join(base_dir, 'semantic_MRs')
    seg_dir = os.path.join(base_dir, 'semantic_labels_only')
    
    if os.path.exists(mri_dir) and os.path.exists(seg_dir):
        mri_files = [f for f in os.listdir(mri_dir) if f.endswith(MRI_SUFFIX)]
        label_files = [f for f in os.listdir(seg_dir) if f.endswith(LABEL_SUFFIX)]
        
        print(f" Dataset Files:")
        print(f"   MRI volumes: {len(mri_files)}")
        print(f"   Label volumes: {len(label_files)}")
        
        # Use modular matching to count pairs
        try:
            volume_pairs = match_mri_label_pairs(mri_dir, seg_dir)
            print(f"    Matched volume pairs: {len(volume_pairs)}")
            if len(volume_pairs) > 0:
                print(f"   Sample: {volume_pairs[0]['patient_week']}")
        except Exception as e:
            print(f"    Error: {e}")
    else:
        print(f"directories not found")


def get_dataset_stats(dataset: HipMRI3DDataset) -> Dict[str, Any]:
    """
    Get comprehensive statistics about a dataset.
    
    Args:
        dataset: HipMRI3DDataset instance
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'num_volumes': len(dataset),
        'target_shape': dataset.target_shape,
        'augment': dataset.augment,
        'normalize': dataset.normalize,
        'normalize_method': getattr(dataset, 'normalize_method', 'zscore'),
        'classes': HIPMRI_CLASSES,
        'num_classes': NUM_CLASSES
    }
    
    if len(dataset) > 0:
        # Get sample info
        sample_info = dataset.volume_pairs[0]
        stats['sample_patient_week'] = sample_info['patient_week']
        stats['mri_dir'] = dataset.mri_dir
        stats['labels_dir'] = dataset.labels_dir
    
    return stats


# ============================================================================
# MAIN AND TESTING
# ============================================================================

if __name__ == "__main__":
    # Example usage
    BASE_DIR = "/home/groups/comp3710/HipMRI_Study_open"
    TARGET_SHAPE = DEFAULT_TARGET_SHAPE
    BATCH_SIZE = 4
    
    print_dataset_info(BASE_DIR)