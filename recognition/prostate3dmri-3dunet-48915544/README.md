# Prostate 3D MRI Segmentation with 3D U-Net

## Overview

This project implements a deep learning solution for 3D prostate MRI segmentation using a 3D U-Net architecture on the HipMRI 3D dataset. The goal is to achieve high Dice similarity coefficients (≥ 0.70) for automated prostate and surrounding organ segmentation.

## Project Structure

```
prostate3dmri-3dunet-48915544/
├── modules.py          # 3D U-Net model definition
├── dataset.py          # HipMRI dataset loader and preprocessing
├── train.py           # Training script with validation
├── predict.py         # Evaluation and inference script
├── utils.py           # Utility functions and helpers
└── README.md          # This documentation
```

## Setup & Dependencies

### Required Packages

- torch>=1.9.0
- torchvision>=0.10.0
- numpy>=1.21.0
- matplotlib>=3.4.0
- scikit-learn>=1.0.0
- nibabel>=3.2.0
- scipy>=1.7.0
- PIL>=8.0.0
- tqdm>=4.62.0

### Dataset Setup

1. Dataset paths:
   - MR volumes: `/home/groups/comp3710/HipMRI_Study_open/semantic_MRs`
   - Labels: `/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only`
2. Expected format: NIfTI files (.nii or .nii.gz)

## Usage

### Training
```bash
python train.py --data_dir /home/groups/comp3710/HipMRI_Study_open
```

### Evaluation
```bash
python predict.py --checkpoint_path ./checkpoints/best.pth
```

## Model Architecture

The 3D U-Net combines:
- Encoder-decoder structure with skip connections
- 3D convolutions for volumetric processing
- Batch normalization and dropout regularization
- Multi-class segmentation: Background, Body Outline, Bone, Bladder, Rectum, Prostate (6 classes)

## License

This project is part of COMP3710 Pattern Analysis coursework.

## References

- mdciri. 3D-augmentation-techniques (GitHub repository). Available at: https://github.com/mdciri/3D-augmentation-techniques (accessed 2025-10-30).
- GeeksforGeeks. How to convert an array of indices to one-hot encoded NumPy array. Available at: https://www.geeksforgeeks.org/numpy/how-to-convert-an-array-of-indices-to-one-hot-encoded-numpy-array/ (accessed 2025-10-30).