import os
import re
import sys
import random
import logging
from typing import List, Tuple

import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModernBatchBalancedDataset:
    """
    Modern implementation of a batch-balanced dataset for OCR training.
    """

    def __init__(self, opt):
        """
        Args:
            opt: Configuration object with dataset parameters.
        """
        self.opt = opt
        self.data_loaders = []
        self.dataset_iterators = []

        # Create log directory if it doesn't exist
        log_dir = f'./saved_models/{opt.experiment_name}'
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = open(f'{log_dir}/log_dataset.txt', 'a', encoding='utf-8')

        self._initialize_datasets()

    def _initialize_datasets(self):
        """Initialize all datasets and their corresponding data loaders."""
        self.log_file.write('-' * 80 + '\n')
        self.log_file.write(f'Dataset Configuration:\n')
        self.log_file.write(f'Train data root: {self.opt.train_data}\n')
        self.log_file.write(f'Select data: {self.opt.select_data}\n')
        self.log_file.write(f'Batch ratio: {self.opt.batch_ratio}\n')
        self.log_file.write('-' * 80 + '\n')

        all_datasets = []
        for dataset_name in self.opt.select_data:
            dataset = self._create_dataset(dataset_name)
            all_datasets.append(dataset)

        # Create a single concatenated dataset
        concatenated_dataset = ConcatDataset(all_datasets)

        # Create a single data loader for the concatenated dataset
        self.data_loader = DataLoader(
            concatenated_dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=int(self.opt.workers),
            collate_fn=ModernAlignCollate(
                imgH=self.opt.imgH,
                imgW=self.opt.imgW,
                keep_ratio_with_pad=self.opt.PAD,
                augmentation=getattr(self.opt, 'augmentation', True),
                rgb=self.opt.rgb
            ),
            pin_memory=True,
            persistent_workers=True if int(self.opt.workers) > 0 else False,
            drop_last=True
        )
        self.dataset_iterator = iter(self.data_loader)
        logger.info(f"Total training samples: {len(concatenated_dataset)}")
        self.log_file.write(f"Total training samples: {len(concatenated_dataset)}\n")

    def _create_dataset(self, dataset_name: str) -> Dataset:
        """Create dataset based on the dataset name/path."""
        dataset_path = os.path.join(self.opt.train_data, dataset_name)

        # Check if it's a CSV-based dataset
        if os.path.isfile(os.path.join(dataset_path, 'labels.csv')):
            logger.info(f"Loading CSV-based dataset from: {dataset_path}")
            return ModernOCRDataset(dataset_path, self.opt)
        else:
            raise ValueError(f"Dataset not found or invalid format for {dataset_path}")

    def get_batch(self) -> Tuple[torch.Tensor, List[str]]:
        """Get a batch from the data loader."""
        try:
            images, labels = next(self.dataset_iterator)
        except StopIteration:
            # Restart iterator
            self.dataset_iterator = iter(self.data_loader)
            images, labels = next(self.dataset_iterator)
        except Exception as e:
            logger.error(f"Error getting batch: {e}")
            raise RuntimeError("Could not retrieve a valid batch.") from e
        return images, labels

    def __len__(self):
        return len(self.data_loader)

    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'log_file'):
            self.log_file.close()


class ModernOCRDataset(Dataset):
    """Modern CSV-based OCR dataset implementation."""

    def __init__(self, root: str, opt):
        self.root = root
        self.opt = opt
        csv_path = os.path.join(root, 'labels.csv')
        try:
            self.df = pd.read_csv(csv_path, sep=',', engine='python', usecols=['filename', 'words'],
                                  keep_default_na=False)
        except Exception as e:
            logger.error(f"Cannot load CSV from {csv_path}: {e}")
            raise

        self.filtered_df = self._filter_samples()
        logger.info(f'CSV Dataset {os.path.basename(root)}: {len(self.filtered_df)}/{len(self.df)} samples loaded.')

    def _filter_samples(self) -> pd.DataFrame:
        """Filter samples based on constraints."""
        df = self.df.copy()
        df = df[df['words'].str.len() <= self.opt.batch_max_length]
        df = df[df['words'].str.len() > 0]

        if not self.opt.data_filtering_off and hasattr(self.opt, 'character'):
            # Create character set from character_list
            valid_chars = set(self.opt.character)
            # Filter out samples with invalid characters
            df = df[df['words'].apply(lambda x: all(c in valid_chars for c in x))]

        return df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.filtered_df)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:
        row = self.filtered_df.iloc[index]
        img_fname = row['filename']
        img_fpath = os.path.join(self.root, img_fname)  # Images are in the same folder
        label = row['words']

        try:
            if self.opt.rgb:
                img = cv2.imread(img_fpath)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.imread(img_fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Cannot load image: {img_fpath}")
        except Exception as e:
            logger.error(f"Error loading image {img_fpath}: {e}")
            # Return a placeholder to be filtered by collate_fn
            return None, ""

        if not getattr(self.opt, 'sensitive', True):
            label = label.lower()

        return img, label


class ModernAlignCollate:
    """Modern collate function with advanced augmentations using Albumentations."""

    def __init__(self, imgH: int = 32, imgW: int = 100, keep_ratio_with_pad: bool = False,
                 augmentation: bool = True, rgb: bool = True):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.augmentation = augmentation
        self.rgb = rgb

        # Common transforms
        if rgb:
            self.normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            self.normalize = A.Normalize(mean=[0.5], std=[0.5])

        # Augmentation pipeline
        if augmentation:
            self.augment = A.Compose([
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=0.4),
                ], p=0.4),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-3, 3), p=0.3),
                self.normalize,
                ToTensorV2()
            ])
        else:
            self.augment = A.Compose([
                self.normalize,
                ToTensorV2()
            ])

    def __call__(self, batch: List[Tuple[np.ndarray, str]]) -> Tuple[torch.Tensor, List[str]]:
        # Filter out None samples from failed __getitem__ calls
        batch = [sample for sample in batch if sample[0] is not None]
        if not batch:
            # Return empty tensors if the whole batch failed
            return torch.Tensor(), []

        images, labels = zip(*batch)
        processed_images = []

        for img in images:
            # Handle grayscale conversion if needed
            if not self.rgb and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif self.rgb and len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            if self.keep_ratio_with_pad:
                h, w = img.shape[:2]
                ratio = w / h
                new_w = int(ratio * self.imgH)

                # Resize maintaining aspect ratio
                img_resized = cv2.resize(img, (new_w, self.imgH))

                # Apply augmentations
                augmented = self.augment(image=img_resized)['image']

                # Pad to target width
                c, h, w = augmented.shape
                pad_width = self.imgW - w
                if pad_width > 0:
                    augmented = TF.pad(augmented, [0, 0, pad_width, 0], fill=0)
                elif pad_width < 0:  # If wider than target, resize again
                    augmented = TF.resize(augmented, [self.imgH, self.imgW])

                processed_images.append(augmented)
            else:
                # Resize directly to target size
                img_resized = cv2.resize(img, (self.imgW, self.imgH))
                augmented = self.augment(image=img_resized)['image']
                processed_images.append(augmented)

        image_tensors = torch.stack(processed_images)
        return image_tensors, list(labels)


# Legacy compatibility functions
def hierarchical_dataset(root, opt):
    """Create a dataset from the root directory."""
    return ModernOCRDataset(root, opt), ""


# Legacy compatibility aliases
Batch_Balanced_Dataset = ModernBatchBalancedDataset
AlignCollate = ModernAlignCollate