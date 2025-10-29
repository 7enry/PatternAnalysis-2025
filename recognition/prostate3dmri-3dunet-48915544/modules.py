"""
3D U-Net Model for Prostate 3D MRI Segmentation
================================================

This module contains the 3D U-Net architecture implementation for volumetric prostate MRI segmentation.
The model includes encoder-decoder structure with skip connections for 3D medical image segmentation.

Author: Henry
Course: COMP3710 Pattern Analysis
"""

import torch
import torch.nn as nn
from typing import Optional


class UNet3D(nn.Module):
    """
    3D U-Net for prostate MRI segmentation with encoder-decoder architecture.
    
    Architecture: Encoder downsamples features → Bottleneck → Decoder upsamples with skip connections
    Classes: Background, Body Outline, Bone, Bladder, Rectum, Prostate (6 total)
    
    Args:
        in_channels: Input channels (1 for grayscale MRI)
        out_channels: Output classes (6 for prostate anatomy) 
        init_features: Base feature count (32)
        dropout_rate: Regularization rate (0.1)
    """
    def __init__(self, in_channels=1, out_channels=6, init_features=32, dropout_rate=0.1):
        """Initialize 3D U-Net with encoder-decoder structure."""
        super(UNet3D, self).__init__()

        features = init_features
        
        # Encoder: 5 levels, each doubles features, halves spatial dimensions
        self.encoder1 = UNet3D._block(in_channels, features)           # 32 features
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2)          # 64 features  
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4)      # 128 features
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8)      # 256 features
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder5 = UNet3D._block(features * 8, features * 16)     # 512 features
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck: deepest features, highest abstraction
        self.bottleneck = UNet3D._block(features * 16, features * 32)  # 1024 features
        self.dropout = nn.Dropout3d(dropout_rate)  # Regularization

        # Decoder: 5 levels, transpose conv upsamples, skip connections restore details
        self.upconv5 = nn.ConvTranspose3d(features * 32, features * 16, kernel_size=2, stride=2)
        self.decoder5 = UNet3D._block(features * 16 * 2, features * 16)  # Skip: enc5
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet3D._block(features * 8 * 2, features * 8)    # Skip: enc4
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet3D._block(features * 4 * 2, features * 4)    # Skip: enc3
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet3D._block(features * 2 * 2, features * 2)    # Skip: enc2
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet3D._block(features * 2, features)            # Skip: enc1

        # Final 1x1 conv: maps features to class logits
        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass: encoder → bottleneck → decoder with skip connections.
        
        Args:
            x: Input volume (B, C, D, H, W)
        Returns:
            Logits (B, num_classes, D, H, W) - no softmax applied
        """
        # Encoder: extract hierarchical features, downsample spatial dims
        enc1 = self.encoder1(x)                    # 32 features, full resolution
        enc2 = self.encoder2(self.pool1(enc1))     # 64 features, 1/2 resolution
        enc3 = self.encoder3(self.pool2(enc2))     # 128 features, 1/4 resolution
        enc4 = self.encoder4(self.pool3(enc3))     # 256 features, 1/8 resolution
        enc5 = self.encoder5(self.pool4(enc4))     # 512 features, 1/16 resolution

        # Bottleneck: highest abstraction level
        bottleneck = self.bottleneck(self.pool5(enc5))  # 1024 features, 1/32 resolution
        bottleneck = self.dropout(bottleneck)           # Apply dropout for regularization

        # Decoder: upsample and restore spatial resolution using skip connections
        dec5 = self.upconv5(bottleneck)            # Upsample 2x
        dec5 = torch.cat((dec5, enc5), dim=1)      # Skip connection: restore fine details
        dec5 = self.decoder5(dec5)                 # Process combined features
        
        dec4 = self.upconv4(dec5)                  # Upsample 2x
        dec4 = torch.cat((dec4, enc4), dim=1)      # Skip connection
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)                  # Upsample 2x
        dec3 = torch.cat((dec3, enc3), dim=1)      # Skip connection
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)                  # Upsample 2x
        dec2 = torch.cat((dec2, enc2), dim=1)      # Skip connection
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)                  # Upsample 2x
        dec1 = torch.cat((dec1, enc1), dim=1)      # Skip connection
        dec1 = self.decoder1(dec1)                 # Final decoder block

        # Output: 1x1 conv maps features to class logits (no softmax)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features):
        """
        Standard U-Net block: Conv3d → BatchNorm → ReLU → Conv3d → BatchNorm → ReLU.
        
        Args:
            in_channels: Input channels
            features: Output features
        Returns:
            Sequential block with 2 conv layers
        """
        return nn.Sequential(
            # First conv: 3x3x3 kernel, padding=1 preserves spatial dims
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=features),  # Normalize features
            nn.ReLU(inplace=True),                  # Non-linearity
            # Second conv: same params, processes normalized features
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=features),  # Normalize again
            nn.ReLU(inplace=True)                   # Final activation
        )
    
    def get_model_summary(self) -> str:
        """Get model architecture summary with parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return f"""
3D U-Net Model Summary:
Input channels: {self.conv.in_channels if hasattr(self.conv, 'in_channels') else 'N/A'}
Output channels: {self.conv.out_channels}
Initial features: {32}
Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}
        """

if __name__ == "__main__":
    # Test the model
    model = UNet3D(in_channels=1, out_channels=6, init_features=32)
    print(model.get_model_summary())