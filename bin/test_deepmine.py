#!/usr/bin/env python

import os
import sys
import time
import yaml
import json
import argparse
import numpy as np
from kaldiio import WriteHelper

sys.path.append('./')
import torch
import torch.nn.functional as F
import tools.utils as utils
from models.rnn_apc import make_model as make_se
from models.rnn_asr import make_model
from tools.speech_loader import SpeechDataset, SpeechDataLoader

class Config():
    name = 'config'

def main():
    parser = argparse.ArgumentParser(description="Configuration for training an APC model")
   
    parser.add_argument("--test_config")
    parser.add_argument("--use_cmvn", default=False, action='store_true', help="Use cmvn or not")
    parser.add_argument("--batch_size", default=32, type=int, help="Training minibatch size")
    parser.add_argument("--load_data_workers", default=2, type=int, help="Number of parallel data loaders")
    parser.add_argument("--resume_model", default='', type=str, help="Use cmvn or not")
    parser.add_argument("--print_freq", default=100, type=int, help="Number of iter to print")
    parser.add_argument("--out_prob", type=str, help="output file to store phrase id log prob")
    parser.add_argument("--out_embedding", type=str, help="name of output embedding ark and scp file")
    parser.add_argument("--seed", default=1, type=int, help="random number seed")

    args = parser.parse_args()
    with open(args.test_config) as f:
        config = yaml.safe_load(f)

    config['path_test'] = [j for i, j in config['test_data'].items()]
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

    enc_args = Config()
    for key, val in args.encoder.items():
        setattr(enc_args, key, val)
    shared_encoder = make_se(enc_args.input_size, enc_args.N, enc_args.N_embed, enc_args.d_model, enc_args.d_ff, enc_args.h, enc_args.dropout)
        
    dec_args = Config()
    for key, val in args.asr_decoder.items():
        setattr(dec_args, key, val)
    dec_args.d_model = enc_args.d_model
    model = make_model(shared_encoder, dec_args)

    if args.resume_model:
        resume_model = torch.load(args.resume_model, map_location='cpu')
        model.load_state_dict(resume_model)

    num_params = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
    print("Number of parameters: {}".format(num_params))
    
    if use_cuda:
        model = model.cuda()

    testset = SpeechDataset(args.path_test, args.left_ctx, args.right_ctx, args.skip_frame)
    if args.use_cmvn:
        testset._load_cmvn(args.global_cmvn)
    test_loader = SpeechDataLoader(testset, args.batch_size, num_workers=args.load_data_workers, shuffle=False)
    print("Finish Loading test files. Number batches: {}".format(len(test_loader)))
    
    batch_time = utils.AverageMeter('Time', ':6.3f')
    progress = utils.ProgressMeter(len(test_loader), batch_time)
    end = time.time()
    
    ark_writer = WriteHelper('ark,scp:{}.ark,{}.scp'.format(args.out_embedding, args.out_embedding))
    prob_writer = open(args.out_prob, 'w')

    with torch.no_grad():
        model.eval()
        for i, data in enumerate(test_loader):
            utt_list, feats, _, feat_sizes, _, _, _ = data
            batch_size, mask_size, _ = feats.size()
            feat_sizes /= 2
            
            if args.use_gpu:
                feats = feats.cuda()
                feat_sizes = feat_sizes.cuda()
            
            _, phrase_out, t_embedding = model(feats, feat_sizes)
            
            logprob = F.log_softmax(phrase_out, dim=-1)
            for j in range(len(utt_list)):
                ark_writer(utt_list[j], t_embedding[j].cpu().numpy())
                prob_writer.write(utt_list[j]+' '+str(logprob[j].cpu().numpy())+'\n')
            
            batch_time.update(time.time() - end)
            if i % args.print_freq == 0:
                progress.print(i)

def subsequent_mask(size):
    ret = torch.ones(size, size, dtype=torch.uint8)
    return torch.tril(ret, out=ret).unsqueeze(0)

    
if __name__ == '__main__':
    main()


