import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor,  ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class ModernizedOCRModel(nn.Module):
    """
    Modernized OCR model with an improved, configuration-driven architecture.

    This model combines transformation, feature extraction, sequence modeling,
    and prediction stages for end-to-end text recognition.
    """

    def __init__(self, config: Dict[str, Any]):
        super(ModernizedOCRModel, self).__init__()
        self.config = config
        self.stages = {
            'Trans': config.get('Transformation', 'None'),
            'Feat': config.get('FeatureExtraction', 'ResNet'),
            'Seq': config.get('SequenceModeling', 'BiLSTM'),
            'Pred': config.get('Prediction', 'Attn')
        }

        # Initialize modules
        self._init_transformation()
        self._init_feature_extraction()
        self._init_sequence_modeling()
        self._init_prediction()

    def _init_transformation(self) -> None:
        """Initialize the transformation module."""
        if self.stages['Trans'] == 'TPS':
            self.transformation = TPS_SpatialTransformerNetwork(
                F=self.config.get('num_fiducial', 20),
                I_size=(self.config.get('imgH', 32), self.config.get('imgW', 100)),
                I_r_size=(self.config.get('imgH', 32), self.config.get('imgW', 100)),
                I_channel_num=self.config.get('input_channel', 1)
            )
        else:
            self.transformation = nn.Identity()

    def _init_feature_extraction(self) -> None:
        """Initialize the feature extraction module."""
        input_channel = self.config.get('input_channel', 1)
        output_channel = self.config.get('output_channel', 512)

        if self.stages['Feat'] == 'VGG':
            self.feature_extraction = VGG_FeatureExtractor(input_channel, output_channel)
        elif self.stages['Feat'] == 'ResNet':
            self.feature_extraction = ResNet_FeatureExtractor(input_channel, output_channel)
        else:
            raise ValueError(f"Unsupported FeatureExtraction: {self.stages['Feat']}")

        self.feature_extraction_output = getattr(self.feature_extraction, 'output_channel', output_channel)
        # Adaptive pooling to handle variations in feature map height
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 1))

    def _init_sequence_modeling(self) -> None:
        """Initialize the sequence modeling module."""
        if self.stages['Seq'] == 'BiLSTM':
            hidden_size = self.config.get('hidden_size', 256)
            self.sequence_modeling = nn.Sequential(
                BidirectionalLSTM(self.feature_extraction_output, hidden_size, hidden_size),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
            )
            self.sequence_modeling_output = hidden_size
        else:
            self.sequence_modeling = nn.Identity()
            self.sequence_modeling_output = self.feature_extraction_output

    def _init_prediction(self) -> None:
        """Initialize the prediction module."""
        num_class = self.config.get('num_class', 37)

        if self.stages['Pred'] == 'CTC':
            self.prediction = nn.Linear(self.sequence_modeling_output, num_class)
        elif self.stages['Pred'] == 'Attn':
            hidden_size = self.config.get('hidden_size', 256)
            self.prediction = Attention(self.sequence_modeling_output, hidden_size, num_class)
        else:
            raise ValueError(f"Unsupported Prediction method: {self.stages['Pred']}")

    def forward(self,
                image: torch.Tensor,
                text: Optional[torch.Tensor] = None,
                is_train: bool = True) -> torch.Tensor:
        """
        Forward pass of the OCR model.

        Args:
            image: Input image tensor of shape (B, C, H, W).
            text: Ground truth text tensor for training (required for Attention).
            is_train: Whether in training mode.

        Returns:
            Model predictions.
        """
        # Transformation stage
        transformed_image = self.transformation(image)

        # Feature extraction stage
        visual_feature = self.feature_extraction(transformed_image)

        # Reshape features for sequence modeling: (B, C, H, W) -> (B, W, C)
        visual_feature = self.adaptive_pool(visual_feature.permute(0, 3, 1, 2))  # -> (B, W, C, 1)
        visual_feature = visual_feature.squeeze(3)  # -> (B, W, C)

        # Sequence modeling stage
        contextual_feature = self.sequence_modeling(visual_feature)

        # Prediction stage
        if self.stages['Pred'] == 'CTC':
            prediction = self.prediction(contextual_feature.contiguous())
        else:  # Attention-based prediction
            batch_max_length = self.config.get('batch_max_length', 25)
            prediction = self.prediction(
                contextual_feature.contiguous(),
                text,
                is_train,
                batch_max_length=batch_max_length
            )

        return prediction
