"""
Model module for Pneumonia Detection.
Implements transfer learning with EfficientNet-B0 (better accuracy than ResNet50).
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import NUM_CLASSES, GRADCAM_LAYER


class PneumoniaModelEfficientNet(nn.Module):
    """
    Pneumonia Detection Model using EfficientNet-B0.
    Better accuracy and efficiency than ResNet50.
    """
    
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(PneumoniaModelEfficientNet, self).__init__()
        
        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Get the number of features in the last layer
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        
        # Store target layer for Grad-CAM (last conv layer)
        self.target_layer = 'features'
        
    def _freeze_backbone(self):
        """Freeze all layers except the classifier."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        
        # Unfreeze the classifier
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classifier."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def get_target_layer(self):
        """Get the target layer for Grad-CAM."""
        # Return the last convolutional block
        return self.backbone.features[-1]


class PneumoniaModel(nn.Module):
    """
    Pneumonia Detection Model using transfer learning.
    Based on ResNet50 with custom classification head.
    
    Args:
        num_classes: Number of output classes (default: 2)
        pretrained: Whether to use pretrained weights (default: True)
        freeze_backbone: Whether to freeze backbone layers (default: False)
    """
    
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(PneumoniaModel, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Get the number of features in the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        
        # Store target layer for Grad-CAM
        self.target_layer = GRADCAM_LAYER
        
    def _freeze_backbone(self):
        """Freeze all layers except the final classification layer."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze the final fc layer
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the final classification layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        # Forward through all layers except fc
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def get_target_layer(self):
        """Get the target layer for Grad-CAM visualization."""
        return dict(self.backbone.named_modules())[self.target_layer]


def create_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    device: str = 'cuda',
    model_type: str = 'efficientnet'
) -> nn.Module:
    """
    Create and initialize the model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        device: Device to load model on
        model_type: 'efficientnet' or 'resnet50'
        
    Returns:
        Initialized model
    """
    if model_type == 'efficientnet':
        model = PneumoniaModelEfficientNet(num_classes=num_classes, pretrained=pretrained)
        print(f"Using EfficientNet-B0 model")
    else:
        model = PneumoniaModel(num_classes=num_classes, pretrained=pretrained)
        print(f"Using ResNet50 model")
    
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = create_model(num_classes=2, pretrained=True, device=device)
    
    # Count parameters
    total, trainable = count_parameters(model)
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)
    output = model(test_input)
    
    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output}")
    
    # Test feature extraction
    features = model.get_features(test_input)
    print(f"\nFeatures shape: {features.shape}")
