import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ModernVGGFeatureExtractor(nn.Module):
    """
    Modernized VGG-based feature extractor with improved design patterns.
    Based on the CRNN architecture with enhancements for better performance.
    """

    def __init__(self, input_channels: int, output_channels: int = 512,
                 dropout_rate: float = 0.1, use_attention: bool = False):
        super().__init__()

        self.channel_progression = [
            output_channels // 8,  # 64
            output_channels // 4,  # 128
            output_channels // 2,  # 256
            output_channels  # 512
        ]

        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.output_channel = output_channels

        # Feature extraction backbone
        self.backbone = self._build_backbone(input_channels)

        # Optional attention mechanism
        if use_attention:
            self.attention = SpatialAttention(output_channels)

        # Initialize weights
        self._initialize_weights()

    def _build_backbone(self, input_channels: int) -> nn.Sequential:
        """Build the main feature extraction backbone."""
        layers = []
        # Block 1
        layers.extend([
            nn.Conv2d(input_channels, self.channel_progression[0], 3, 1, 1),
            nn.BatchNorm2d(self.channel_progression[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/2, W/2
        ])
        # Block 2
        layers.extend([
            nn.Conv2d(self.channel_progression[0], self.channel_progression[1], 3, 1, 1),
            nn.BatchNorm2d(self.channel_progression[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/4, W/4
        ])
        # Block 3
        layers.extend([
            nn.Conv2d(self.channel_progression[1], self.channel_progression[2], 3, 1, 1),
            nn.BatchNorm2d(self.channel_progression[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_progression[2], self.channel_progression[2], 3, 1, 1),
            nn.BatchNorm2d(self.channel_progression[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/8, W/4
        ])
        # Block 4
        layers.extend([
            nn.Conv2d(self.channel_progression[2], self.channel_progression[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_progression[3]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(self.channel_progression[3], self.channel_progression[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_progression[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/16, W/4
        ])
        # Final compression
        layers.extend([
            nn.Conv2d(self.channel_progression[3], self.channel_progression[3], 2, 1, 0),
            nn.BatchNorm2d(self.channel_progression[3]),
            nn.ReLU(inplace=True)
        ])
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using modern best practices."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional attention."""
        features = self.backbone(x)
        if self.use_attention:
            features = self.attention(features)
        return features


class ModernResNetFeatureExtractor(nn.Module):
    """
    Modernized ResNet-based feature extractor with architectural improvements.
    """

    def __init__(self, input_channels: int, output_channels: int = 512,
                 dropout_rate: float = 0.1, use_se: bool = True):
        super().__init__()
        self.use_se = use_se
        layers = [1, 2, 5, 3]  # ResNet configuration
        self.output_channel = output_channels

        self.backbone = ModernResNet(
            input_channels=input_channels,
            output_channels=output_channels,
            block=ModernBasicBlock,
            layers=layers,
            dropout_rate=dropout_rate,
            use_se=use_se
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)



class ModernBasicBlock(nn.Module):
    """Modernized BasicBlock with optional SE attention and improved design."""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, dropout_rate: float = 0.1,
                 use_se: bool = True):
        super().__init__()
        # Main convolution path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)

        # Squeeze-and-Excitation
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        # Residual connection
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Main path
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)


class ModernResNet(nn.Module):
    """Modernized ResNet architecture for OCR feature extraction."""

    def __init__(self, input_channels: int, output_channels: int,
                 block: nn.Module, layers: List[int], dropout_rate: float = 0.1,
                 use_se: bool = True):
        super().__init__()
        self.channel_progression = [
            output_channels // 4,  # 128
            output_channels // 2,  # 256
            output_channels,  # 512
            output_channels  # 512
        ]
        self.in_channels = output_channels // 8  # 64
        self.dropout_rate = dropout_rate
        self.use_se = use_se

        self.stem = self._build_stem(input_channels)
        self.layer1 = self._make_layer(block, self.channel_progression[0], layers[0])
        self.layer2 = self._make_layer(block, self.channel_progression[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.channel_progression[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.channel_progression[3], layers[3], stride=2)

        self._initialize_weights()

    def _build_stem(self, input_channels: int) -> nn.Sequential:
        """Build the initial stem network."""
        return nn.Sequential(
            nn.Conv2d(input_channels, self.in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )

    def _make_layer(self, block: nn.Module, channels: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """Create a layer with multiple blocks."""
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )
        layers = [block(self.in_channels, channels, stride, downsample,
                        self.dropout_rate, self.use_se)]
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels, dropout_rate=self.dropout_rate,
                                use_se=self.use_se))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using modern best practices."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        se = self.global_pool(x).view(b, c)
        se = self.fc(se).view(b, c, 1, 1)
        return x * se.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for feature maps."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention


# --- Legacy Class Names for Compatibility ---
VGG_FeatureExtractor = ModernVGGFeatureExtractor
ResNet_FeatureExtractor = ModernResNetFeatureExtractor
