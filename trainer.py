#!/usr/bin/env python3
"""
OCR Model Training Script
Converted from Jupyter notebook
"""

import os
import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict
import pandas as pd


def get_config(file_path):
    """Load and process configuration from YAML file"""
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)

    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python',
                             usecols=['filename', 'words'], keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character = ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char

    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt


def main():
    """Main training function"""
    # Set CUDNN settings for performance
    cudnn.benchmark = True
    cudnn.deterministic = False

    # Load configuration
    config_file = "config_files/en_filtered_config.yaml"
    opt = get_config(config_file)

    # Start training
    print(f"Starting training with config: {config_file}")
    try:
        train(opt, amp=False)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()