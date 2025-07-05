import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class TPS_SpatialTransformerNetwork(nn.Module):
    """
    Modernized Thin Plate Spline (TPS) Spatial Transformer Network.

    This module learns to apply a TPS transformation to the input image to
    rectify text, making it easier for the recognition model to process.
    """

    def __init__(self, F: int, I_size: Tuple[int, int], I_r_size: Tuple[int, int], I_channel_num: int = 1):
        """
        Initialize the TPS-STN module.

        Args:
            F: Number of fiducial points.
            I_size: Size of the input image (height, width).
            I_r_size: Size of the rectified output image (height, width).
            I_channel_num: Number of channels in the input image.
        """
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size
        self.I_channel_num = I_channel_num

        # Localization network to predict fiducial points
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)

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
        # Predict the fiducial points
        C_prime = self.LocalizationNetwork(I)  # (batch_size, F, 2)

        # Generate the sampling grid from the predicted points
        build_P_prime = self.GridGenerator.build_P_prime(C_prime)  # (batch_size, H*W, 2)

        # Reshape the grid for grid_sample
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])

        # Apply the transformation to the input image
        I_r = F.grid_sample(I, build_P_prime_reshape, padding_mode='border', align_corners=True)

        return I_r


class LocalizationNetwork(nn.Module):
    """
    A network to predict the locations of fiducial points for TPS.
    """

    def __init__(self, F: int, I_channel_num: int):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.I_channel_num = I_channel_num

        # Backbone for feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.I_channel_num, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x32 -> 32x16
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 32x16 -> 16x8
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 16x8 -> 8x4
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )

        # Regressor for fiducial points
        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.F * 2)

        # Initialize weights for the regressor
        self.localization_fc2.weight.data.fill_(0)
        # Initialize the bias with a reference grid for stability
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.ones(int(F / 2)) * -1.0
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, batch_I: torch.Tensor) -> torch.Tensor:
        """
        Predict fiducial points from the input image.
        """
        features = self.conv(batch_I).view(batch_I.size(0), -1)
        fc1_output = self.localization_fc1(features)
        C_prime = self.localization_fc2(fc1_output).view(batch_I.size(0), self.F, 2)
        return C_prime


class GridGenerator(nn.Module):
    """
    Generates the sampling grid for the TPS transformation.
    """

    def __init__(self, F: int, I_r_size: Tuple[int, int]):
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)  # (F, 2)
        self.P = self._build_P(self.I_r_width, self.I_r_height)  # (H*W, 3)

        # Precompute parts of the TPS matrix inverse
        self.register_buffer('inv_delta_C', torch.tensor(self._build_inv_delta_C(self.F, self.C)).float())  # (F+3, F+3)
        self.register_buffer('P_hat', torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float())  # (H*W, F+3)

    def _build_C(self, F: int) -> np.ndarray:
        """
        Return a reference grid of fiducial points C.
        """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        return np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)

    def _build_P(self, W: int, H: int) -> np.ndarray:
        """
        Return a regular grid P.
        """
        x_coords = np.linspace(-1.0, 1.0, W)
        y_coords = np.linspace(-1.0, 1.0, H)
        X, Y = np.meshgrid(x_coords, y_coords)
        P = np.stack([X.flatten(), Y.flatten()], axis=1)
        return np.concatenate([np.ones((W * H, 1)), P], axis=1)  # (H*W, 3)

    def _build_inv_delta_C(self, F: int, C: np.ndarray) -> np.ndarray:
        """
        Return the inverse of the matrix delta_C.
        """
        R = np.linalg.norm(C[np.newaxis, :, :] - C[:, np.newaxis, :], axis=2) ** 2
        R[R == 0] = self.eps  # for numerical stability
        R = R * np.log(R)

        P = np.concatenate([np.ones((F, 1)), C], axis=1)
        delta_C = np.concatenate([
            np.concatenate([P, R], axis=1),
            np.concatenate([np.zeros((3, 3)), P.T], axis=1)
        ], axis=0)

        return np.linalg.inv(delta_C)

    def _build_P_hat(self, F: int, C: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Return the matrix P_hat.
        """
        n = P.shape[0]
        P_tile = np.tile(P, (1, F))  # (H*W, 3*F)
        C_tile = np.tile(C, (n, 1))  # (H*W*F, 2)
        P_diff = P_tile.reshape(n * F, 3) - np.concatenate([np.zeros((n * F, 1)), C_tile], axis=1)
        R = np.linalg.norm(P_diff[:, 1:], axis=1) ** 2
        R[R == 0] = self.eps
        R = R * np.log(R)

        P_hat = np.concatenate([
            np.ones((n, 1)), P[:, 1:], R.reshape(n, F)
        ], axis=1)
        return P_hat

    def build_P_prime(self, C_prime: torch.Tensor) -> torch.Tensor:
        """
        Build the transformed grid P_prime.
        """
        batch_size = C_prime.size(0)
        C_prime_with_zeros = torch.cat((C_prime, torch.zeros(batch_size, 3, 2).float().to(C_prime.device)), dim=1)

        # Calculate transformation parameters T
        T = torch.bmm(self.inv_delta_C.unsqueeze(0).repeat(batch_size, 1, 1), C_prime_with_zeros)

        # Calculate the transformed grid
        P_prime = torch.bmm(self.P_hat.unsqueeze(0).repeat(batch_size, 1, 1), T)
        return P_prime  # (batch_size, H*W, 2)
