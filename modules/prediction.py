import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class Attention(nn.Module):
    """
    Modernized attention-based decoder for text recognition.

    Improvements:
    - Type hints for better code clarity.
    - Proper device handling without a hardcoded device.
    - More efficient tensor operations.
    - Better parameter validation.
    - Cleaner code structure.
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char: torch.Tensor, onehot_dim: int) -> torch.Tensor:
        """
        Convert character indices to one-hot vectors using a modern PyTorch approach.
        """
        return F.one_hot(input_char, num_classes=onehot_dim).float()

    def _initialize_hidden_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state with proper device handling."""
        return (
            torch.zeros(batch_size, self.hidden_size, device=device),
            torch.zeros(batch_size, self.hidden_size, device=device)
        )

    def forward(self, batch_H: torch.Tensor, text: Optional[torch.Tensor] = None,
                is_train: bool = True, batch_max_length: int = 25) -> torch.Tensor:
        """
        Forward pass of the attention-based decoder.

        Args:
            batch_H: Feature map from encoder [batch_size, seq_len, input_size].
            text: Ground truth text for teacher forcing [batch_size, max_length].
            is_train: Whether in training mode.
            batch_max_length: Maximum sequence length.

        Returns:
            Prediction probability distribution [batch_size, num_steps, num_classes].
        """
        batch_size = batch_H.size(0)
        device = batch_H.device
        num_steps = batch_max_length + 1  # +1 for [EOS] token

        # Initialize states
        hidden = self._initialize_hidden_state(batch_size, device)
        output_hiddens = torch.zeros(batch_size, num_steps, self.hidden_size, device=device)

        if is_train:
            if text is None:
                raise ValueError("Ground truth text must be provided during training for teacher forcing.")

            # Teacher forcing mode
            for i in range(num_steps):
                char_onehot = self._char_to_onehot(text[:, i], self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehot)
                output_hiddens[:, i, :] = hidden[0]

            probs = self.generator(output_hiddens)
        else:
            # Inference mode with autoregressive generation
            targets = torch.zeros(batch_size, dtype=torch.long, device=device)  # Start with [GO] token

            for i in range(num_steps):
                char_onehot = self._char_to_onehot(targets, self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehot)
                output_hiddens[:, i, :] = hidden[0]

                # Get next character prediction
                probs_step = self.generator(hidden[0])
                targets = probs_step.argmax(dim=1)

            probs = self.generator(output_hiddens)

        return probs


class AttentionCell(nn.Module):
    """
    Modernized attention cell with improved efficiency and readability.
    """
    def __init__(self, input_size: int, hidden_size: int, num_embeddings: int):
        super(AttentionCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_embeddings = num_embeddings

        # Attention mechanism components
        self.feature_projection = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden_projection = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attention_score = nn.Linear(hidden_size, 1, bias=False)

        # LSTM cell for context integration
        self.lstm_cell = nn.LSTMCell(input_size + num_embeddings, hidden_size)

    def forward(self, prev_hidden: Tuple[torch.Tensor, torch.Tensor],
                batch_H: torch.Tensor, char_onehots: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass of the attention cell.

        Args:
            prev_hidden: Previous hidden state tuple (h, c).
            batch_H: Encoder feature map [batch_size, seq_len, input_size].
            char_onehots: One-hot encoded character vectors [batch_size, num_embeddings].

        Returns:
            Tuple of (new_hidden_state, attention_weights).
        """
        prev_h, prev_c = prev_hidden
        batch_size, seq_len, _ = batch_H.size()

        # Project features and hidden state for attention computation
        feature_proj = self.feature_projection(batch_H)  # [batch_size, seq_len, hidden_size]
        hidden_proj = self.hidden_projection(prev_h).unsqueeze(1)  # [batch_size, 1, hidden_size]

        # Compute attention energies
        attention_energies = self.attention_score(
            torch.tanh(feature_proj + hidden_proj.expand(-1, seq_len, -1))
        )  # [batch_size, seq_len, 1]

        # Calculate attention weights
        alpha = F.softmax(attention_energies.squeeze(-1), dim=1)  # [batch_size, seq_len]

        # Compute context vector
        context = torch.bmm(alpha.unsqueeze(1), batch_H).squeeze(1)  # [batch_size, input_size]

        # Concatenate context with character embedding
        lstm_input = torch.cat([context, char_onehots], dim=1)

        # Update hidden state
        new_hidden = self.lstm_cell(lstm_input, (prev_h, prev_c))

        return new_hidden, alpha
