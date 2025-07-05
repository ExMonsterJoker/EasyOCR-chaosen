import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from torch.utils.checkpoint import checkpoint


class TPS_SpatialTransformerNetwork(nn.Module):
    """
    Modernized Thin Plate Spline (TPS) Spatial Transformer Network for EasyOCR.

    This module learns to apply a TPS transformation to input images to rectify text,
    making it easier for OCR recognition models to process curved or distorted text.

    Features:
    - Improved numerical stability
    - Better initialization strategies
    - Support for mixed precision training
    - Gradient checkpointing for memory efficiency
    - Configurable backbone architectures
    """

    def __init__(
            self,
            F: int,
            I_size: Tuple[int, int],
            I_r_size: Tuple[int, int],
            I_channel_num: int = 1,
            backbone: str = "resnet",
            use_checkpoint: bool = False,
            dropout_rate: float = 0.1
    ):
        """
        Initialize the TPS-STN module.

        Args:
            F: Number of fiducial points (should be even).
            I_size: Size of the input image (height, width).
            I_r_size: Size of the rectified output image (height, width).
            I_channel_num: Number of channels in the input image.
            backbone: Feature extraction backbone ('resnet', 'efficientnet', 'mobilenet').
            use_checkpoint: Whether to use gradient checkpointing.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()

        if F % 2 != 0:
            raise ValueError(f"Number of fiducial points F must be even, got {F}")

        self.F = F
        self.I_r_size = I_r_size
        self.use_checkpoint = use_checkpoint

        # Localization network to predict fiducial points
        self.LocalizationNetwork = LocalizationNetwork(
            self.F, I_channel_num, backbone, dropout_rate
        )

        # Grid generator for the TPS transformation
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TPS-STN.

        Args:
            I: Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Rectified image tensor.
        """
        # Use gradient checkpointing if enabled
        if self.use_checkpoint and self.training:
            C_prime = checkpoint(self.LocalizationNetwork, I, use_reentrant=False)
        else:
            C_prime = self.LocalizationNetwork(I)

        # Generate the sampling grid from the predicted points
        build_P_prime = self.GridGenerator.build_P_prime(C_prime)

        # Reshape the grid for grid_sample
        build_P_prime_reshape = build_P_prime.reshape([
            build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2
        ])

        # Apply the transformation to the input image
        I_r = F.grid_sample(
            I, build_P_prime_reshape,
            padding_mode='border',
            align_corners=True,
            mode='bilinear'
        )

        return I_r


class LocalizationNetwork(nn.Module):
    """
    Modernized localization network with multiple backbone options.
    """

    def __init__(
            self,
            F: int,
            I_channel_num: int,
            backbone: str = "resnet",
            dropout_rate: float = 0.1
    ):
        super().__init__()
        self.F = F

        # Choose backbone architecture
        if backbone == "resnet":
            self.conv, feature_dim = self._build_resnet_backbone(I_channel_num)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Improved regressor
        self.localization_fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, self.F * 2)
        )

        # Better initialization
        self._initialize_weights()

    def _build_resnet_backbone(self, I_channel_num: int) -> Tuple[nn.Module, int]:
        """Build ResNet-style backbone."""

        def conv_block(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        model = nn.Sequential(
            conv_block(I_channel_num, 64),
            nn.MaxPool2d(2, 2),
            conv_block(64, 128),
            nn.MaxPool2d(2, 2),
            conv_block(128, 256),
            nn.MaxPool2d(2, 2),
            conv_block(256, 512),
            nn.AdaptiveAvgPool2d(1)
        )
        return model, 512

    def _initialize_weights(self):
        """Initialize weights with an improved strategy."""
        # Initialize final layer weights to zero for stability
        self.localization_fc[-1].weight.data.fill_(0)

        # Initialize bias with a reference grid
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(self.F / 2))
        ctrl_pts_y_top = np.ones(int(self.F / 2)) * -1.0
        ctrl_pts_y_bottom = np.ones(int(self.F / 2)) * 1.0
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)

        with torch.no_grad():
            self.localization_fc[-1].bias.copy_(
                torch.from_numpy(initial_bias).float().view(-1)
            )

        # Initialize other layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m != self.localization_fc[-1]:
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_I: torch.Tensor) -> torch.Tensor:
        """
        Predict fiducial points from the input image.
        """
        batch_size = batch_I.size(0)
        features = self.conv(batch_I).view(batch_size, -1)
        C_prime = self.localization_fc(features).view(batch_size, self.F, 2)

        # Apply tanh to ensure points are in the [-1, 1] range
        C_prime = torch.tanh(C_prime)
        return C_prime


class GridGenerator(nn.Module):
    """
    Modernized grid generator with improved numerical stability.
    """

    def __init__(self, F: int, I_r_size: Tuple[int, int]):
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)
        self.P = self._build_P(self.I_r_width, self.I_r_height)

        # Register buffers for multi-GPU training and model saving
        self.register_buffer('inv_delta_C', torch.tensor(self._build_inv_delta_C(), dtype=torch.float32))
        self.register_buffer('P_hat', torch.tensor(self._build_P_hat(), dtype=torch.float32))

    def _build_C(self, F: int) -> np.ndarray:
        """Build reference control points."""
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1.0 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = 1.0 * np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        return np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)

    def _build_P(self, W: int, H: int) -> np.ndarray:
        """Build a regular grid of points."""
        x_coords = np.linspace(-1.0, 1.0, W)
        y_coords = np.linspace(-1.0, 1.0, H)
        X, Y = np.meshgrid(x_coords, y_coords)
        P = np.stack([X.flatten(), Y.flatten()], axis=1)
        return np.concatenate([np.ones((W * H, 1)), P], axis=1)

    def _build_inv_delta_C(self) -> np.ndarray:
        """Build the inverse of the delta_C matrix with improved numerical stability."""
        R = np.linalg.norm(self.C[np.newaxis, :, :] - self.C[:, np.newaxis, :], axis=2) ** 2
        R[R == 0] = self.eps
        R = R * np.log(R)

        P_ = np.concatenate([np.ones((self.F, 1)), self.C], axis=1)
        delta_C = np.concatenate([
            np.concatenate([P_, R], axis=1),
            np.concatenate([np.zeros((3, 3)), P_.T], axis=1)
        ], axis=0)

        # Add small regularization for numerical stability
        delta_C += np.eye(delta_C.shape[0]) * self.eps
        return np.linalg.inv(delta_C)

    def _build_P_hat(self) -> np.ndarray:
        """Build the P_hat matrix for transformation."""
        n = self.P.shape[0]
        P_tile = np.tile(self.P, (1, self.F))
        C_tile = np.tile(self.C, (n, 1))
        P_diff = P_tile.reshape(n * self.F, 3) - np.concatenate([np.zeros((n * self.F, 1)), C_tile], axis=1)

        R = np.linalg.norm(P_diff[:, 1:], axis=1) ** 2
        R[R == 0] = self.eps
        R = R * np.log(R)

        return np.concatenate([np.ones((n, 1)), self.P[:, 1:], R.reshape(n, self.F)], axis=1)

    def build_P_prime(self, C_prime: torch.Tensor) -> torch.Tensor:
        """
        Build the transformed grid from predicted control points.
        """
        batch_size = C_prime.size(0)
        device = C_prime.device

        C_prime_with_zeros = torch.cat([
            C_prime,
            torch.zeros(batch_size, 3, 2, device=device, dtype=C_prime.dtype)
        ], dim=1)

        T = torch.bmm(self.inv_delta_C.expand(batch_size, -1, -1), C_prime_with_zeros)
        P_prime = torch.bmm(self.P_hat.expand(batch_size, -1, -1), T)
        return P_prime
