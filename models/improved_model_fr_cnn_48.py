import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual_block import ResidualBlock


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ImprovedNet48(nn.Module):
    """
    Optimized ResNet for 48x48 facial expression recognition

    Architecture designed specifically for small input size:
    - Input: 48x48x3
    - Lighter initial layers (no aggressive downsampling)
    - 3 residual layers instead of 4
    - Reduced channel counts for faster inference
    - Optimized for real-time performance
    """
    def __init__(self, num_classes=7):
        super().__init__()

        # Initial conv - smaller kernel and no downsampling for 48x48 input
        # 48x48x3 -> 48x48x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Light pooling to reduce spatial dimensions
        # 48x48x32 -> 24x24x32
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Residual blocks with gradual downsampling
        # Layer 1: 24x24x32 -> 24x24x64
        self.layer1 = self._make_layer(32, 64, 2, stride=1)

        # Layer 2: 24x24x64 -> 12x12x128
        self.layer2 = self._make_layer(64, 128, 2, stride=2)

        # Layer 3: 12x12x128 -> 6x6x256
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        # Global average pooling: 6x6x256 -> 1x1x256
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier with dropout for regularization
        # 256 -> 128 -> 7
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # Initialize weights using He initialization
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer with multiple residual blocks"""
        layers = []
        # First block may downsample
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # Remaining blocks maintain dimensions
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU networks"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)       # 48x48x3  -> 48x48x32
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # 48x48x32 -> 24x24x32

        # Residual layers
        x = self.layer1(x)      # 24x24x32  -> 24x24x64
        x = self.layer2(x)      # 24x24x64  -> 12x12x128
        x = self.layer3(x)      # 12x12x128 -> 6x6x256

        # Global pooling and classification
        x = self.avgpool(x)     # 6x6x256 -> 1x1x256
        x = torch.flatten(x, 1) # 1x1x256 -> 256
        x = self.fc(x)          # 256 -> 7

        return x

    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# Test function to verify model architecture
def test_model():
    """Test model with sample input"""
    model = ImprovedNet48(num_classes=7)

    # Create dummy input (batch_size=1, channels=3, height=48, width=48)
    dummy_input = torch.randn(1, 3, 48, 48)

    # Forward pass
    output = model(dummy_input)

    # Print model info
    total_params, trainable_params = model.count_parameters()
    print("=" * 70)
    print("ImprovedNet48 - Architecture Test")
    print("=" * 70)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 70)

    # Verify output shape
    assert output.shape == (1, 7), f"Expected output shape (1, 7), got {output.shape}"
    print("✓ Model test passed!")

    return model


if __name__ == "__main__":
    test_model()
