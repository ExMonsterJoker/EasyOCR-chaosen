import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """
    Converts between text-label and text-index for CTC loss.
    """

    def __init__(self, character):
        self.character = list(character)
        # Note: 0 is reserved for the blank token
        self.dict = {char: i + 1 for i, char in enumerate(self.character)}
        self.num_classes = len(self.character) + 1  # +1 for blank token

    def encode(self, text: List[str], batch_max_length: int = 25) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes a list of text strings into tensors for CTC loss.
        - [GO] token is not used in CTC.
        - Blank token (0) is used for padding.
        """
        lengths = [len(s) for s in text]
        text_indices = torch.zeros((len(text), batch_max_length), dtype=torch.long)
        for i, t in enumerate(text):
            if len(t) > batch_max_length:
                t = t[:batch_max_length]

            indices = [self.dict.get(char.lower(), 0) for char in t]
            text_indices[i, :len(indices)] = torch.LongTensor(indices)

        return text_indices, torch.IntTensor(lengths)

    def decode(self, text_index: torch.Tensor, length: torch.Tensor) -> List[str]:
        """
        Decodes CTC-encoded predictions into text strings.
        """
        texts = []
        for i, l in enumerate(length):
            t = text_index[i, :]
            char_list = []
            for j in range(l):
                # Remove consecutive duplicates and blank tokens
                if t[j] != 0 and (j == 0 or t[j] != t[j - 1]):
                    char_list.append(self.character[t[j] - 1])
            texts.append(''.join(char_list))
        return texts


class AttnLabelConverter(object):
    """
    Converts between text-label and text-index for Attention-based models.
    """

    def __init__(self, character: str):
        # Define special tokens
        self.GO = '[GO]'
        self.EOS = '[s]'

        self.character = [self.GO, self.EOS] + list(character)
        self.dict = {char: i for i, char in enumerate(self.character)}
        self.num_classes = len(self.character)

    def encode(self, text: List[str], batch_max_length: int = 25) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes text strings into tensors for Attention loss.
        Adds [GO] and [s] tokens.
        """
        # +2 for [GO] and [s]
        lengths = [len(s) + 2 for s in text]
        # Padded with [GO] token (index 0)
        batch_text = torch.zeros((len(text), batch_max_length + 2), dtype=torch.long)

        for i, t in enumerate(text):
            if len(t) > batch_max_length:
                t = t[:batch_max_length]

            text_sequence = [self.dict.get(char.lower(), self.dict[self.GO]) for char in t]
            text_sequence = [self.dict[self.GO]] + text_sequence + [self.dict[self.EOS]]
            batch_text[i, :len(text_sequence)] = torch.LongTensor(text_sequence)

        return batch_text, torch.IntTensor(lengths)

    def decode(self, text_index: torch.Tensor, length: torch.Tensor) -> List[str]:
        """
        Decodes Attention-encoded predictions into text strings.
        """
        texts = []
        for i, l in enumerate(length):
            char_list = []
            for j in range(l):
                char_idx = text_index[i, j].item()
                if char_idx == self.dict[self.EOS]:
                    break  # Stop at end-of-sequence token
                if char_idx != self.dict[self.GO]:
                    char_list.append(self.character[char_idx])
            texts.append(''.join(char_list))
        return texts


class Averager(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add(self, val: Union[torch.Tensor, float, int]):
        if isinstance(val, torch.Tensor):
            self.val = val.item()
            self.sum += val.item()
        else:
            self.val = val
            self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


# --- General Purpose Utilities ---

class AttrDict(dict):
    """Dictionary with attribute-style access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    # For Apple Silicon
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
