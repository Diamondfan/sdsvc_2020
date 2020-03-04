#!/usr/bin/env python

import os
import sys
import time
import yaml
import json
import argparse
import math
import numpy as np

sys.path.append('./')
import torch
import tools.utils as utils
from models.rnn_apc import make_model as make_se
from models.rnn_asr import make_model
from tools.speech_loader import SpeechDataset, SpeechDataLoader

class Config():
    name = 'config'

def main():
    parser = argparse.ArgumentParser(description="Configuration for training an APC model")
   
    parser.add_argument("--exp_dir")
    parser.add_argument("--train_config")
    parser.add_argument("--data_config")
    parser.add_argument("--use_cmvn", default=False, action='store_true', help="Use cmvn or not")
    parser.add_argument("--batch_size", default=32, type=int, help="Training minibatch size")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="Initial learning rate")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--load_data_workers", default=2, type=int, help="Number of parallel data loaders")
    parser.add_argument("--anneal_lr_ratio", default=0.5, type=float, help="learning rate decay ratio")
    parser.add_argument("--anneal_lr_epoch", default=10, type=int, help="epoch to decay learning rate")
    parser.add_argument("--phrase_lambda", default=0, type=float, help="The ratio of pid task")
    parser.add_argument("--resume_encoder", default='', type=str, help="Use cmvn or not")
    parser.add_argument("--fix_encoder", default=False, action='store_true', help="Use cmvn or not")
    parser.add_argument("--print_freq", default=100, type=int, help="Number of iter to print")
    parser.add_argument("--seed", default=1, type=int, help="random number seed")

    args = parser.parse_args()
    with open(args.train_config) as f:
        config = yaml.safe_load(f)

    with open(args.data_config) as f:
        data = yaml.safe_load(f)
        config['path_train'] = [j for i, j in data['train_data_path'].items()]
        config['path_dev'] = [j for i, j in data['dev_data_path'].items()]
        config['global_cmvn'] = data['global_cmvn']

    for key, val in config.items():
        setattr(args, key, val)
    for var in vars(args):
        config[var] = getattr(args, var)

    print("Experiment starts with config {}".format(json.dumps(config, sort_keys=True, indent=4)))

    use_cuda = args.use_gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    enc_args = Config()
    for key, val in args.encoder.items():
        setattr(enc_args, key, val)
    shared_encoder = make_se(enc_args.input_size, enc_args.N, enc_args.N_embed, enc_args.d_model, enc_args.d_ff, enc_args.h, enc_args.dropout)
    if args.resume_encoder:
        print("Loading model from {}".format(args.resume_encoder))
        resume_encoder = torch.load(args.resume_encoder, map_location='cpu')
        shared_encoder.load_state_dict(resume_encoder)
        
    dec_args = Config()
    for key, val in args.asr_decoder.items():
        setattr(dec_args, key, val)
    dec_args.d_model = enc_args.d_model
    model = make_model(shared_encoder, dec_args)

    num_params = 0
    for name, param in model.named_parameters():
        if args.fix_encoder and name.startswith('ctc_net.shared_encoder'):
            param.requires_grad=False
        num_params += param.numel()
    print("Number of parameters: {}".format(num_params))

    criterion_ctc = torch.nn.CTCLoss(reduction='mean')
    criterion_phrase = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    if use_cuda:
        model = model.cuda()

    trainset = SpeechDataset(args.path_train, args.left_ctx, args.right_ctx, args.skip_frame)
    if args.use_cmvn:
        trainset._load_cmvn(args.global_cmvn)

    train_loader = SpeechDataLoader(trainset, args.batch_size, num_workers=args.load_data_workers, shuffle=True)
    print("Finish Loading training files. Number batches: {}".format(len(train_loader)))

    validset = SpeechDataset(args.path_dev, args.left_ctx, args.right_ctx, args.skip_frame)
    if args.use_cmvn:
        validset._load_cmvn(args.global_cmvn)

    valid_loader = SpeechDataLoader(validset, args.batch_size, num_workers=args.load_data_workers, shuffle=False)
    print("Finish Loading dev files. Number batches: {}".format(len(valid_loader)))
    best_per = 100
    for epoch in range(args.epochs):
        if epoch > args.anneal_lr_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.anneal_lr_ratio
        print("Learning rate: {:.4e}".format(optimizer.param_groups[0]['lr']), flush=True)
        
        model.train()
        train_loss, train_per = run_epoch(epoch, train_loader, model, criterion_ctc, criterion_phrase, args, optimizer, is_train=True)
        model.eval()
        with torch.no_grad():
            valid_loss, valid_per = run_epoch(epoch, valid_loader, model, criterion_ctc, criterion_phrase, args, is_train=False)

        print("Epoch {} done, Train Loss: {:.4f}, Train PER: {:.4f} Valid Loss: {:.4f} Valid PER: {:.4f}".format(epoch, train_loss, train_per, valid_loss, valid_per), flush=True) 
        
        output_file=args.exp_dir + '/model.' + str(epoch) + '.mdl'
        torch.save(model.state_dict(), output_file)

        if valid_per < best_per:
            best_per = valid_per
            output_file=args.exp_dir + '/best.mdl'
            torch.save(model.state_dict(), output_file)

def subsequent_mask(size):
    ret = torch.ones(size, size, dtype=torch.uint8)
    return torch.tril(ret, out=ret).unsqueeze(0)

def run_epoch(epoch, dataloader, model, criterion_ctc, criterion_phrase, args, optimizer=None, is_train=True):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    ctc_losses = utils.AverageMeter('CtcLoss', ":.4e")
    phrase_losses = utils.AverageMeter('PIDLoss', ":.4e")
    phone_wers = utils.AverageMeter('Phone_WER', ':.4f')
    pid_accs = utils.AverageMeter('PID_Acc', ':.4f')
    progress = utils.ProgressMeter(len(dataloader), batch_time, losses, ctc_losses, phone_wers, phrase_losses, pid_accs, prefix="Epoch: [{}]".format(epoch))
    
    end = time.time()
    
    for i, data in enumerate(dataloader):
        utt_list, feats, labels, feat_sizes, label_sizes, phrase_label, _ = data
        batch_size, mask_size, _ = feats.size()
        feat_sizes /= 2
        #uni_mask = (feats != 0)[:,:,0].unsqueeze(-2).byte() & subsequent_mask(mask_size)
 
        if args.use_gpu:
            feats = feats.cuda()
            #uni_mask = uni_mask.cuda()
            labels = labels.cuda()
            feat_sizes = feat_sizes.cuda()
            label_sizes = label_sizes.cuda()
            phrase_label = phrase_label.cuda()
        
        ctc_out, phrase_out, _ = model(feats, feat_sizes)

        ctc_loss = criterion_ctc(ctc_out.transpose(0,1), labels, feat_sizes, label_sizes)
        phrase_loss = criterion_phrase(phrase_out, phrase_label.view(-1))
        loss = ctc_loss + args.phrase_lambda * phrase_loss
        batch_errs, batch_tokens = model.compute_wer(torch.max(ctc_out, dim=-1)[1].cpu().numpy(), feat_sizes.cpu().numpy(), labels.cpu().numpy(), label_sizes.cpu().numpy())
        
        phone_wers.update(batch_errs/batch_tokens, batch_tokens)
        correct = torch.sum(torch.argmax(phrase_out, dim=-1).view(-1) == phrase_label.view(-1)).item()
        batch_num = torch.sum((phrase_label != -1)).item() + 0.0000001
        pid_accs.update(correct/batch_num, batch_num)
        
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses.update(loss.item(), batch_size)
        ctc_losses.update(ctc_loss.item(), batch_size)
        phrase_losses.update(phrase_loss.item(), batch_size)
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            progress.print(i)
    return losses.avg, phone_wers.avg
    
if __name__ == '__main__':
    main()


