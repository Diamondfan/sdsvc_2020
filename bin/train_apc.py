#!/usr/bin/env python

import os
import sys
import time
import yaml
import json
import argparse
import numpy as np

sys.path.append('./')
import torch
import tools.utils as utils
from models.tsfm_apc import make_model
from tools.speech_loader import SpeechDataset, SpeechDataLoader


def main():
    parser = argparse.ArgumentParser(description="Configuration for training an APC model")
   
    parser.add_argument("--exp_dir")
    parser.add_argument("--train_config")
    parser.add_argument("--data_config")
    parser.add_argument("--batch_size", default=32, type=int, help="Training minibatch size")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="Initial learning rate")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--time_shift", default=1, type=int, help="Given f_{t}, predict f_{t + n}, where n is the time_shift")
    parser.add_argument("--load_data_workers", default=2, type=int, help="Number of parallel data loaders")
    parser.add_argument("--anneal_lr_ratio", default=0.5, type=float, help="learning rate decay ratio")
    parser.add_argument("--anneal_lr_epoch", default=10, type=int, help="epoch to decay learning rate")
    parser.add_argument("--print_freq", default=100, type=int, help="Number of iter to print")
    parser.add_argument("--seed", default=1234, type=int, help="random number seed")



    args = parser.parse_args()
    with open(args.train_config) as f:
        config = yaml.safe_load(f)

    with open(args.data_config) as f:
        data = yaml.safe_load(f)
        config['path_train'] = [j for i, j in data['train_data_path'].items()]
        config['path_dev'] = [j for i, j in data['dev_data_path'].items()]

    for key, val in config.items():
        setattr(args, key, val)

    print("Experiment starts with config {}".format(json.dumps(config, sort_keys=True, indent=4)))

    use_cuda = args.use_gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    model = make_model(args.input_size, args.N, args.N_embed, args.d_model, args.d_ff, args.h, args.dropout)
    
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("Number of parameters: {}".format(num_params))

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if use_cuda:
        model = model.cuda()

    trainset = SpeechDataset(args.path_train, args.left_ctx, args.right_ctx, args.skip_frame)
    train_loader = SpeechDataLoader(trainset, args.batch_size, num_workers=args.load_data_workers, shuffle=True)
    print("Finish Loading training files...")

    validset = SpeechDataset(args.path_dev, args.left_ctx, args.right_ctx, args.skip_frame)
    valid_loader = SpeechDataLoader(validset, args.batch_size, num_workers=args.load_data_workers, shuffle=False)
    print("Finish Loading dev files...")
    best_acc = 100
    for epoch in range(args.epochs):
        if epoch > args.anneal_lr_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.anneal_lr_ratio
        print("Learning rate: {:.4e}".format(optimizer.param_groups[0]['lr']), flush=True)
        
        model.train()
        train_loss = run_epoch(epoch, train_loader, model, criterion, args, optimizer, is_train=True)

        model.eval()
        with torch.no_grad():
            valid_loss = run_epoch(epoch, valid_loader, model, criterion, args, is_train=False)

        print("Epoch {} done, Train Loss: {:.4f}, Valid Loss: {:.4f}".format(epoch, train_loss, valid_loss), flush=True) 
        
        output_file=args.exp_dir + '/model.' + str(epoch) + '.mdl'
        torch.save(model.state_dict(), output_file)

        if valid_loss < best_acc:
            best_acc = valid_loss
            output_file=args.exp_dir + '/best.mdl'
            torch.save(model.state_dict(), output_file)

def subsequent_mask(size):
    ret = torch.ones(size, size, dtype=torch.uint8)
    return torch.tril(ret, out=ret).unsqueeze(0)

def run_epoch(epoch, dataloader, model, criterion, args, optimizer=None, is_train=True):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    progress = utils.ProgressMeter(len(dataloader), batch_time, losses, prefix="Epoch: [{}]".format(epoch))
    
    time_shift = args.time_shift
    end = time.time()

    for i, data in enumerate(dataloader):
        utt_list, feats, _, _, _ = data
        mask_size = feats.size(1)-time_shift
        uni_mask = (feats[:,:-time_shift,:] != 0)[:,:,0].unsqueeze(-2).byte() & subsequent_mask(mask_size)
        
        if args.use_gpu:
            feats = feats.cuda()
            uni_mask = uni_mask.cuda()
        
        outputs = model(feats[:, :-time_shift, :], uni_mask)
        #outputs = model(feats[:, :-time_shift, :])

        loss = criterion(outputs, feats[:, time_shift:, :])
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses.update(loss.item(), feats.size(0))
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            progress.print(i)
    return losses.avg
    
if __name__ == '__main__':
    main()


