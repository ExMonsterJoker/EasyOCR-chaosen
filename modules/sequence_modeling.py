import torch
import torch.nn as nn
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class SequenceModelingBase(ABC, nn.Module):
    """Abstract base class for sequence modeling modules."""

    @abstractmethod
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Forward pass for sequence modeling."""
        pass

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Returns the output feature size."""
        pass


class BidirectionalLSTM(SequenceModelingBase):
    """
    Modernized Bidirectional LSTM for sequence modeling in OCR.

    Features:
    - Improved parameter handling and initialization.
    - Support for multiple layers and dropout.
    - Better gradient flow with proper parameter flattening.
    - Configurable architecture with sensible defaults.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 bias: bool = True,
                 batch_first: bool = True):
        """
        Initialize the BidirectionalLSTM module.

        Args:
            input_size: Number of expected features in the input.
            hidden_size: Number of features in the hidden state.
            output_size: Number of features in the output.
            num_layers: Number of recurrent layers.
            dropout: Dropout probability for hidden states (if num_layers > 1).
            bias: Whether to use bias parameters.
            batch_first: Whether input tensors are batch-first.
        """
        super(BidirectionalLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self._output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first

        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bias=bias,
            batch_first=batch_first,
            bidirectional=True
        )

        # Linear projection layer
        self.linear = nn.Linear(hidden_size * 2, output_size, bias=bias)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize LSTM and linear layer weights using Xavier initialization."""
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
                # Set forget gate bias to 1 for better gradient flow at the beginning of training
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

        # Initialize linear layer
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    @property
    def output_size(self) -> int:
        """Returns the output feature size."""
        return self._output_size

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BidirectionalLSTM.

        Args:
            input_features: Input sequence features.
                            Shape: (batch_size, seq_len, input_size) if batch_first=True.

        Returns:
            torch.Tensor: Contextual features.
                          Shape: (batch_size, seq_len, output_size) if batch_first=True.
        """
        # Flatten parameters for better performance (handles DataParallel)
        self._flatten_parameters()

        # Forward pass through LSTM
        lstm_out, _ = self.rnn(input_features)

        # Apply linear transformation
        output = self.linear(lstm_out)

        return output

    def _flatten_parameters(self):
        """Safely flatten LSTM parameters for better performance."""
        try:
            self.rnn.flatten_parameters()
        except (RuntimeError, AttributeError):
            # This is not critical, so we just continue
            pass


class TransformerSequenceModel(SequenceModelingBase):
    """
    Modern Transformer-based sequence modeling as an alternative to LSTM.
    Provides better parallelization and often superior performance for OCR tasks.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize the Transformer sequence model.

        Args:
            input_size: Number of input features.
            hidden_size: Hidden dimension size.
            output_size: Number of output features.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            activation: Activation function ('relu' or 'gelu').
        """
        super(TransformerSequenceModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self._output_size = output_size

        # Input projection if needed
        self.input_projection = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(hidden_size, output_size)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize transformer weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @property
    def output_size(self) -> int:
        """Returns the output feature size."""
        return self._output_size

    def forward(self, input_features: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Transformer sequence model.

        Args:
            input_features: Input sequence features (batch_size, seq_len, input_size).
            src_key_padding_mask: Padding mask for variable length sequences.

        Returns:
            torch.Tensor: Contextual features (batch_size, seq_len, output_size).
        """
        # Project input to hidden dimension
        x = self.input_projection(input_features)

        # Apply transformer
        transformer_out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Project to output dimension
        output = self.output_projection(transformer_out)

        return output
