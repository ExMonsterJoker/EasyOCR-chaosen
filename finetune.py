import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import os

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation

def finetune(opt):
    """ Fine-tune a pre-trained model on new dataset """
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Dataset preparation
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    
    train_dataset = Batch_Balanced_Dataset(opt)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        pin_memory=True)
    
    # Model setup
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    model = Model(opt)
    
    # Load pre-trained weights
    if not os.path.exists(opt.pretrained_model):
        raise FileNotFoundError(f"Pre-trained model not found: {opt.pretrained_model}")
    
    print(f'Loading pre-trained model from {opt.pretrained_model}')
    pretrained_dict = torch.load(opt.pretrained_model, map_location=device)
    model_dict = model.state_dict()
    
    # Handle different number of classes
    pretrained_num_class = pretrained_dict['Prediction.generator.weight'].size(0) if 'Prediction.generator.weight' in pretrained_dict else None
    
    if pretrained_num_class and pretrained_num_class != opt.num_class:
        print(f'Class number changed: {pretrained_num_class} -> {opt.num_class}')
        print('Excluding prediction layer from pre-trained weights')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                         if not k.startswith('Prediction')}
    
    # Load weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    print('Pre-trained weights loaded successfully!')
    
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    
    # Loss and optimizer (start fresh)
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    
    # Use lower learning rate for fine-tuning
    finetune_lr = opt.lr * 0.1  # 10x smaller learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr, betas=(opt.beta1, 0.999))
    
    print(f'Fine-tuning with learning rate: {finetune_lr}')
    
    # Training loop
    start_iter = 0
    best_accuracy = -1
    
    for i in range(start_iter, opt.num_iter):
        for p in model.parameters():
            p.requires_grad = True
        
        # Training step
        cpu_images, cpu_texts = train_dataset.get_batch()
        image = cpu_images.to(device)
        text, length = converter.encode(cpu_texts, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)
        
        if 'CTC' in opt.Prediction:
            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.log_softmax(2).permute(1, 0, 2)
            cost = criterion(preds, text, preds_size, length)
        else:
            preds = model(image, text[:, :-1])
            target = text[:, 1:]
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        
        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        
        # Validation
        if i % opt.valInterval == 0:
            elapsed_time = time.time() - start_time
            print(f'[{i}/{opt.num_iter}] Loss: {cost:0.5f} elapsed_time: {elapsed_time:0.5f}')
            
            with torch.no_grad():
                valid_loss, current_accuracy, _, _, _ = validation(
                    model, criterion, valid_loader, converter, opt)
            
            print(f'Valid loss: {valid_loss:0.5f}, accuracy: {current_accuracy:0.3f}')
            
            # Save best model
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                torch.save(model.state_dict(), 
                          f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                print(f'Best accuracy: {best_accuracy:0.3f}')
            
            model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True, help='Where to store logs and models')
    parser.add_argument('--pretrained_model', required=True, help='Path to pre-trained model')
    parser.add_argument('--train_data', required=True, help='Path to training dataset')
    parser.add_argument('--valid_data', required=True, help='Path to validation dataset')
    parser.add_argument('--select_data', type=str, default='MJ-ST', help='Select training data')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5', help='Assign ratio for each selected data')
    parser.add_argument('--batch_size', type=int, default=192, help='Input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='Number of iterations')
    parser.add_argument('--valInterval', type=int, default=2000, help='Validation interval')
    parser.add_argument('--lr', type=float, default=1, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--grad_clip', type=float, default=5, help='Gradient clipping')
    
    # Model architecture
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='Feature extraction stage')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='Sequence modeling stage')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage')
    
    # Add other necessary arguments...
    
    opt = parser.parse_args()
    
    # Create experiment directory
    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)
    
    finetune(opt)
