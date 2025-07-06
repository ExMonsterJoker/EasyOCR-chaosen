import os
import yaml
import time
import random
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Tuple, Union
from utils import CTCLabelConverter, AttnLabelConverter, Averager, AttrDict
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import ModernizedOCRModel as Model
from test import validation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> AttrDict:
    """Load configuration from a YAML file."""
    def dict_to_attrdict(d):
        """Recursively convert nested dictionaries to AttrDict objects."""
        if isinstance(d, dict):
            return AttrDict({k: dict_to_attrdict(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_attrdict(item) for item in d]
        else:
            return d
    
    # Corrected to specify UTF-8 encoding to prevent UnicodeDecodeError on Windows.
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return dict_to_attrdict(config_dict)


def setup_device_and_seed(config: AttrDict):
    """Set up device and random seeds for reproducibility."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        cudnn.benchmark = True
        cudnn.deterministic = True
    logger.info(f"Using device: {device}")
    return device


def setup_data_loaders(config: AttrDict) -> Tuple[Batch_Balanced_Dataset, torch.utils.data.DataLoader, dict]:
    """Set up training and validation data loaders."""
    # Create a temporary config object for the dataset module
    # This is needed because the original dataset classes expect an opt object
    data_opt = AttrDict({
        'experiment_name': config.experiment_name,  # Added this line to fix the error
        'train_data': config.data.train_data,
        'select_data': config.data.select_data,
        'batch_ratio': config.data.batch_ratio,
        'total_data_usage_ratio': config.data.total_data_usage_ratio,
        'batch_size': config.training.batch_size,
        'workers': config.data.workers,
        'imgH': config.data.imgH,
        'imgW': config.data.imgW,
        'rgb': config.data.rgb,
        'character': config.data.character_list,
        'batch_max_length': config.data.batch_max_length,
        'data_filtering_off': config.data.data_filtering_off,
        'PAD': config.data.PAD,
        'Transformation': config.model.Transformation.name,
        'FeatureExtraction': config.model.FeatureExtraction.name,
        'SequenceModeling': config.model.SequenceModeling.name,
        'Prediction': config.model.Prediction.name,
        'num_fiducial': config.model.Transformation.num_fiducial,
        'input_channel': 3 if config.data.rgb else 1,
        'output_channel': config.model.FeatureExtraction.output_channel,
        'hidden_size': config.model.SequenceModeling.hidden_size,
    })

    train_dataset = Batch_Balanced_Dataset(data_opt)

    AlignCollate_valid = AlignCollate(imgH=data_opt.imgH, imgW=data_opt.imgW, keep_ratio_with_pad=data_opt.PAD)
    valid_dataset, _ = hierarchical_dataset(root=config.data.valid_data, opt=data_opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.training.batch_size,
        shuffle=True, num_workers=int(config.data.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(valid_dataset)}")

    return train_dataset, valid_loader, data_opt


def setup_model(config: AttrDict, data_opt: AttrDict, device: torch.device) -> Tuple[
    nn.Module, Union[CTCLabelConverter, AttnLabelConverter]]:
    """Set up the OCR model."""
    if 'CTC' in config.model.Prediction.name:
        converter = CTCLabelConverter(config.data.character_list)
    else:
        converter = AttnLabelConverter(config.data.character_list)

    data_opt.num_class = len(converter.character)

    model = Model(data_opt).to(device)

    # Load pretrained model if specified
    if config.saved_model:
        logger.info(f"Loading model from {config.saved_model}")
        model.load_state_dict(torch.load(config.saved_model, map_location=device), strict=not config.training.fine_tune)

    return model, converter


def train_one_epoch(model, criterion, optimizer, scaler, train_dataset, device, converter, config, writer,
                    current_iter):
    """Train the model for one epoch."""
    model.train()
    loss_avg = Averager()

    num_batches = len(train_dataset) // config.training.batch_size
    pbar = tqdm(range(num_batches), desc=f"Training Iter {current_iter}-{current_iter + num_batches}", leave=False)

    for i in pbar:
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=config.data.batch_max_length)
        text = text.to(device)
        length = length.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=config.training.amp):
            if 'CTC' in config.model.Prediction.name:
                preds = model(image, text)
                preds_size = torch.IntTensor([preds.size(1)] * image.size(0)).to(device)
                cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text, preds_size, length)
            else:
                preds = model(image, text[:, :-1])  # teacher forcing
                target = text[:, 1:]
                cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        scaler.scale(cost).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_avg.add(cost)
        pbar.set_postfix(loss=f"{loss_avg.val:.4f}")
        writer.add_scalar('Loss/train_iter', loss_avg.val, current_iter + i)

    return current_iter + num_batches


def main(config_path: str):
    """Main training pipeline."""
    config = load_config(config_path)
    device = setup_device_and_seed(config)

    # Setup directories and logging
    exp_dir = Path('./saved_models') / config.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=exp_dir / 'runs')

    # Setup data, model, and converter
    train_dataset, valid_loader, data_opt = setup_data_loaders(config)
    model, converter = setup_model(config, data_opt, device)

    # Setup loss function
    if 'CTC' in config.model.Prediction.name:
        criterion = nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token

    # Setup optimizer
    opt_config = config.training.optimizer
    if opt_config.type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt_config.lr, betas=(opt_config.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=opt_config.lr, rho=opt_config.rho, eps=opt_config.eps)

    # Setup scheduler
    scheduler = None
    if config.training.scheduler.use:
        sched_config = config.training.scheduler
        if sched_config.type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_config.step_size, gamma=sched_config.gamma)
        elif sched_config.type == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.num_iter)

    # Setup AMP
    scaler = GradScaler(enabled=config.training.amp)

    # Training loop
    start_iter = 0
    best_accuracy = -1
    best_norm_ED = -1
    current_iter = start_iter

    logger.info(f"Starting training for {config.training.num_iter} iterations.")
    start_time = time.time()

    while current_iter < config.training.num_iter:
        current_iter = train_one_epoch(model, criterion, optimizer, scaler, train_dataset, device, converter, config,
                                       writer, current_iter)

        if current_iter % config.training.valInterval == 0:
            val_loss, current_accuracy, current_norm_ED, _, _, _, _, _ = validation(
                model, criterion, valid_loader, converter, data_opt, device
            )
            writer.add_scalar('Loss/val', val_loss, current_iter)
            writer.add_scalar('Accuracy/val', current_accuracy, current_iter)
            writer.add_scalar('NormED/val', current_norm_ED, current_iter)

            logger.info(
                f"Iter: {current_iter}/{config.training.num_iter} | Val Loss: {val_loss:.4f} | Accuracy: {current_accuracy:.2f}% | Norm ED: {current_norm_ED:.4f}")

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                torch.save(model.state_dict(), exp_dir / 'best_accuracy.pth')
                logger.info(f"New best accuracy model saved at iteration {current_iter}")

            if current_norm_ED > best_norm_ED:
                best_norm_ED = current_norm_ED
                torch.save(model.state_dict(), exp_dir / 'best_norm_ED.pth')

        if scheduler:
            scheduler.step()

    total_time = time.time() - start_time
    logger.info(f"Training finished in {total_time / 3600:.2f} hours.")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file (e.g., config.yaml)')
    args = parser.parse_args()

    main(args.config)
