# OASIS Brain Segmentation with Improved U-Net

## Overview

This project implements a deep learning solution for 2D brain MRI segmentation using an Improved U-Net architecture on the OASIS dataset. The goal is to achieve high Dice similarity coefficients for automated brain tissue segmentation.

## Project Structure

```
oasis-brain-segmentation-48915544/
├── modules.py          # Improved U-Net model definition
├── dataset.py          # OASIS dataset loader and preprocessing
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

### Dataset Setup

1. Place the OASIS dataset in `/home/groups/comp3710/OASIS`
2. Expected directory structure:
   ```
   OASIS/
   ├── images/
   │   ├── OASIS_0001_MR1.nii.gz
   │   └── ...
   └── masks/
       ├── OASIS_0001_MR1_seg.nii.gz
       └── ...
   ```

## Usage

### Training
```bash
python train.py --data_dir /home/groups/comp3710/OASIS
```

### Evaluation
```bash
python predict.py --checkpoint_path ./checkpoints/best_model.pth
```

## Model Architecture

The Improved U-Net combines:
- Encoder-decoder structure with skip connections
- Batch normalization and dropout regularization
- Multi-class segmentation for brain tissues

## License

This project is part of COMP3710 Pattern Analysis coursework.