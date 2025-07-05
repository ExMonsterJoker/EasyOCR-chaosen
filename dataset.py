import os
import re
import sys
import random
import logging
from typing import List, Tuple

import numpy as np
import cv2
import lmdb
import pandas as pd

import torch
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, ImageEnhance

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModernBatchBalancedDataset:
    """
    Modern implementation of a batch-balanced dataset for EasyOCR fine-tuning.
    Supports multiple data sources with configurable ratios and advanced augmentations.
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
                augmentation=getattr(self.opt, 'augmentation', True)
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

        if os.path.isdir(dataset_path) and 'data.mdb' in os.listdir(dataset_path):
            logger.info(f"Loading LMDB dataset from: {dataset_path}")
            return ModernLmdbDataset(dataset_path, self.opt)
        elif os.path.isfile(os.path.join(dataset_path, 'labels.csv')):
            logger.info(f"Loading CSV-based dataset from: {dataset_path}")
            return ModernOCRDataset(dataset_path, self.opt)
        else:
            raise ValueError(f"Unknown or invalid dataset format for {dataset_path}")

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


class ModernLmdbDataset(Dataset):
    """Modern LMDB dataset implementation with better error handling."""

    def __init__(self, root: str, opt):
        self.root = root
        self.opt = opt

        try:
            self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        except lmdb.Error as e:
            logger.error(f'Cannot create LMDB from {root}: {e}')
            raise

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode()).decode())
            self.filtered_index_list = self._filter_samples(txn)

        logger.info(
            f'LMDB Dataset {os.path.basename(root)}: {len(self.filtered_index_list)}/{self.nSamples} samples loaded.')

    def _filter_samples(self, txn) -> List[int]:
        """Filter samples based on length and character constraints."""
        filtered_indices = []
        for index in range(1, self.nSamples + 1):  # LMDB is 1-indexed
            label_key = f'label-{index:09d}'.encode()
            label_bytes = txn.get(label_key)
            if label_bytes is None:
                continue
            label = label_bytes.decode('utf-8')

            if len(label) > self.opt.batch_max_length or len(label) == 0:
                continue

            if not self.opt.data_filtering_off and hasattr(self.opt, 'character'):
                pattern = f'[^{re.escape(self.opt.character)}]'
                if re.search(pattern, label.lower()):
                    continue

            filtered_indices.append(index)
        return filtered_indices

    def __len__(self) -> int:
        return len(self.filtered_index_list)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:
        real_index = self.filtered_index_list[index]
        with self.env.begin(write=False) as txn:
            label_key = f'label-{real_index:09d}'.encode()
            label = txn.get(label_key).decode('utf-8')
            img_key = f'image-{real_index:09d}'.encode()
            imgbuf = txn.get(img_key)

            buf = np.frombuffer(imgbuf, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Cannot decode image at index {real_index} in {self.root}")

            if self.opt.rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if not getattr(self.opt, 'sensitive', True):
            label = label.lower()

        return img, label


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
            pattern = f'[^{re.escape(self.opt.character)}]'
            df = df[~df['words'].str.lower().str.contains(pattern, regex=True)]

        return df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.filtered_df)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:
        row = self.filtered_df.iloc[index]
        img_fname = row['filename']
        img_fpath = os.path.join(self.root, 'images', img_fname)
        label = row['words']

        try:
            if self.opt.rgb:
                img = cv2.imread(img_fpath)
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

    def __init__(self, imgH: int = 32, imgW: int = 100, keep_ratio_with_pad: bool = False, augmentation: bool = True):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.augmentation = augmentation

        # Common transforms
        self.normalize = A.Normalize(mean=[0.5], std=[0.5])

        # Augmentation pipeline
        if augmentation:
            self.augment = A.Compose([
                A.OneOf([
                    A.MotionBlur(blur_limit=5, p=0.5),
                    A.GaussianBlur(blur_limit=5, p=0.5),
                    A.GlassBlur(sigma=0.2, max_delta=2, p=0.3),
                ], p=0.4),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=25, p=0.4),
                ], p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Perspective(scale=(0.02, 0.08), pad_mode=cv2.BORDER_REPLICATE, p=0.4),
                A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-5, 5), shear=(-5, 5), p=0.5),
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
            if self.keep_ratio_with_pad:
                h, w = img.shape[:2]
                ratio = w / h
                new_w = int(ratio * self.imgH)

                # Resize and apply augmentations
                resize_aug = A.Compose([
                    A.Resize(height=self.imgH, width=new_w),
                    self.augment.transforms[0]  # Apply augmentations before padding
                ])
                processed = resize_aug(image=img)['image']

                # Pad to target width
                c, h, w = processed.shape
                pad_width = self.imgW - w
                if pad_width > 0:
                    processed = TF.pad(processed, [0, 0, pad_width, 0], fill=0)
                elif pad_width < 0:  # If wider than target, resize again
                    processed = TF.resize(processed, [self.imgH, self.imgW])

                processed_images.append(processed)
            else:
                # Resize directly to target size and augment
                resize_aug = A.Compose([
                    A.Resize(height=self.imgH, width=self.imgW),
                    self.augment
                ])
                processed = resize_aug(image=img)['image']
                processed_images.append(processed)

        image_tensors = torch.stack(processed_images)
        return image_tensors, list(labels)


# Legacy compatibility functions
def hierarchical_dataset(root, opt):
    return ModernOCRDataset(root, opt), ""  # Simplified for modern use


Batch_Balanced_Dataset = ModernBatchBalancedDataset
AlignCollate = ModernAlignCollate
