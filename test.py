import os
import time
import string
import argparse
import logging
from typing import Tuple, List, Union

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Import a robust edit distance library
try:
    from Levenshtein import distance as edit_distance
except ImportError:
    try:
        from nltk.metrics.distance import edit_distance
    except ImportError:
        import editdistance

        # The editdistance library takes arguments in a different order
        original_edit_distance = editdistance.eval
        edit_distance = lambda s1, s2: original_edit_distance(s1, s2)

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate
from model import ModernizedOCRModel as Model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validation(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        evaluation_loader: torch.utils.data.DataLoader,
        converter: Union[CTCLabelConverter, AttnLabelConverter],
        opt: argparse.Namespace,
        device: torch.device
) -> Tuple[float, float, float, List[str], List[float], List[str], float, int]:
    """
    Modern validation function for OCR models with improved performance and readability.

    Args:
        model: The OCR model to evaluate.
        criterion: Loss function.
        evaluation_loader: DataLoader for validation data.
        converter: Label converter (CTC or Attention).
        opt: Configuration options.
        device: Device for computation.

    Returns:
        Tuple containing:
        - Average validation loss
        - Word-level accuracy (%)
        - Normalized edit distance
        - List of predictions
        - List of confidence scores
        - List of ground truth labels
        - Average inference time per image
        - Total number of samples
    """
    # Initialize metrics
    n_correct = 0
    norm_ED = 0
    total_samples = 0
    total_infer_time = 0
    valid_loss_avg = Averager()

    # Store results
    all_preds, all_labels, all_confidence_scores = [], [], []

    model.eval()
    with torch.no_grad():
        pbar = tqdm(evaluation_loader, desc='Validating', leave=False, ncols=100)
        for image_tensors, labels in pbar:
            batch_size = image_tensors.size(0)
            total_samples += batch_size
            image = image_tensors.to(device, non_blocking=True)

            # Prepare text inputs for loss calculation
            text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)
            text_for_loss = text_for_loss.to(device)
            length_for_loss = length_for_loss.to(device)

            # Measure inference time
            start_time = time.perf_counter()

            # Single forward pass for both loss and prediction
            if 'CTC' in opt.Prediction:
                preds = model(image, None)  # No text needed for CTC inference
                preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)
                cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

                # Decode predictions (greedy)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, preds_size)
            else:  # Attention
                preds = model(image, text_for_loss[:, :-1],
                              is_train=False)  # Use teacher forcing for loss, but decode freely
                target = text_for_loss[:, 1:]  # without [GO] Symbol
                cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

                # Decode predictions
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_loss)

            total_infer_time += (time.perf_counter() - start_time)
            valid_loss_avg.add(cost)

            # Calculate confidence scores
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            # Process each sample in the batch
            for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    gt = gt[:gt.find('[s]')]
                    pred_EOS = pred.find('[s]')
                    if pred_EOS != -1:
                        pred = pred[:pred_EOS]
                        pred_max_prob = pred_max_prob[:pred_EOS]

                if pred == gt:
                    n_correct += 1

                # Calculate Normalized Edit Distance (1 - NED)
                if len(gt) == 0:
                    norm_ED += 1 if len(pred) > 0 else 0
                else:
                    norm_ED += 1 - (edit_distance(pred, gt) / max(len(pred), len(gt)))

                # Store results
                all_preds.append(pred)
                all_labels.append(gt)

                # Calculate confidence score
                confidence_score = pred_max_prob.cumprod(dim=0)[-1].item() if len(pred_max_prob) > 0 else 0.0
                all_confidence_scores.append(confidence_score)

    # Calculate final metrics
    accuracy = n_correct / float(total_samples) * 100
    norm_ED /= float(total_samples)
    avg_infer_time = total_infer_time / float(total_samples)

    logger.info(
        f"Validation Results: Accuracy: {accuracy:.2f}%, NormED: {norm_ED:.4f}, Loss: {valid_loss_avg.val():.4f}")

    return (
        valid_loss_avg.val(),
        accuracy,
        norm_ED,
        all_preds,
        all_confidence_scores,
        all_labels,
        avg_infer_time,
        total_samples
    )
