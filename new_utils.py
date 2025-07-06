import torch
import torch.nn as nn


class CTCLabelConverter(object):
    """
    Converts between text-label and text-index for CTC loss.
    """

    def __init__(self, character):
        self.character = list(character)
        self.dict = {char: i + 1 for i, char in enumerate(self.character)}
        self.num_classes = len(self.character) + 1  # +1 for blank token

    def encode(self, text, batch_max_length=25):
        """
        Encodes a list of text strings into tensors for CTC loss.
        - [GO] token is not used in CTC.
        """
        length = [len(s) for s in text]
        text_indices = torch.IntTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text_indices[i, :len(t)] = torch.IntTensor([self.dict.get(char.lower(), 0) for char in t])
        return text_indices, torch.IntTensor(length)

    def decode(self, text_index, length):
        """
        Decodes CTC-encoded predictions into text strings.
        """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # Remove consecutive duplicates and blanks
                    char_list.append(self.character[t[i] - 1])
            texts.append(''.join(char_list))
        return texts


class AttnLabelConverter(object):
    """
    Converts between text-label and text-index for Attention-based models.
    """

    def __init__(self, character):
        self.character = ['[GO]', '[s]'] + list(character)  # [GO] for start, [s] for end
        self.dict = {char: i for i, char in enumerate(self.character)}
        self.num_classes = len(self.character)

    def encode(self, text, batch_max_length=25):
        """
        Encodes text strings into tensors for Attention loss.
        Adds [GO] and [s] tokens.
        """
        length = [len(s) + 2 for s in text]  # +2 for [GO] and [s]
        batch_text = torch.LongTensor(len(text), batch_max_length + 2).fill_(self.dict['[GO]'])

        for i, t in enumerate(text):
            text_sequence = [self.dict.get(char.lower(), self.dict['[GO]']) for char in t]
            text_sequence = [self.dict['[GO]']] + text_sequence + [self.dict['[s]']]
            batch_text[i, :len(text_sequence)] = torch.LongTensor(text_sequence)

        return batch_text, torch.IntTensor(length)

    def decode(self, text_index, length):
        """
        Decodes Attention-encoded predictions into text strings.
        """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            # Find the end-of-sequence token '[s]'
            eos_pos = text.find('[s]')
            if eos_pos != -1:
                text = text[:eos_pos]
            texts.append(text)
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

    def add(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
