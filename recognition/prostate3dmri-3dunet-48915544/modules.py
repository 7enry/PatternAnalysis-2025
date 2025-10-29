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
    A 3D U-Net model for volumetric prostate MRI segmentation.
    
    Designed for HipMRI dataset with 6 classes:
    - Background (0)
    - Body Outline (1)
    - Bone (2)
    - Bladder (3)
    - Rectum (4)
    - Prostate (5)
    
    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale MRI)
        out_channels (int): Number of output channels/classes (default: 6 for prostate anatomy)
        init_features (int): Number of features in the first layer (default: 32)
        dropout_rate (float): Dropout rate for regularization (default: 0.1)
    """
    def __init__(self, in_channels=1, out_channels=6, init_features=32, dropout_rate=0.1):
        """
        Initializes the 3D U-Net model.
        
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels/classes.
        :param init_features: Number of features in the first layer.
        :param dropout_rate: Dropout rate for regularization.
        """
        super(UNet3D, self).__init__()

        features = init_features
        
        # Encoder path (downsampling)
        self.encoder1 = UNet3D._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder5 = UNet3D._block(features * 8, features * 16)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = UNet3D._block(features * 16, features * 32)
        self.dropout = nn.Dropout3d(dropout_rate)

        # Decoder path (upsampling with skip connections)
        self.upconv5 = nn.ConvTranspose3d(features * 32, features * 16, kernel_size=2, stride=2)
        self.decoder5 = UNet3D._block(features * 16 * 2, features * 16)
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet3D._block(features * 8 * 2, features * 8)
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet3D._block(features * 4 * 2, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet3D._block(features * 2 * 2, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet3D._block(features * 2, features)

        # Final convolutional layer to map features to output channels
        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        """
        Defines the forward pass of the 3D U-Net.
        
        :param x: Input tensor of shape (B, C, D, H, W)
        :return: Output logits of shape (B, num_classes, D, H, W)
        """
        # Encoder path (downsampling)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool5(enc5))
        bottleneck = self.dropout(bottleneck)

        # Decoder path (upsampling with skip connections)
        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5, enc5), dim=1)  # Skip connection
        dec5 = self.decoder5(dec5)
        
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)  # Skip connection
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # Skip connection
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)  # Skip connection
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # Skip connection
        dec1 = self.decoder1(dec1)

        # Final output (logits, softmax applied during loss calculation)
        return self.conv(dec1)