#!/bin/bash

# 

libri_path=/home/vijaysumaravi/Documents/database/librispeech/
stage=3

. ./cmd.sh
. ./path.sh
. parse_options.sh

# Step1. Use librispeech, voxceleb and Deepmine to do unsupervied pretraining.
# We don't extract features for test set at this step.
# Test the code woth only train clean 100h first
if [ $stage -le 0 ]; then
  #format librispeech data as Kaldi data directories
  for part in dev-clean train-clean-100; do
    local/libri_data_prep.sh $libri_path/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done
  #todo: scripts needed to process voxceleb and deepmine dataset
fi

if [ $stage -le 1 ]; then
  # todo: need lexcion to transcribe the words to phoneme sequence
  # Do not need for now.
  echo "1"
fi

if [ $stage -le 2 ]; then
  #feature extraction for librispeech, filter-bank
  fbankdir=data/fbank
  for part in dev_clean train_clean_100; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 10 data/$part exp/make_fbank/$part $fbankdir  
  done
  
  # put training scp to on scp to caculate the cmvn
  if [ -f $fbankdir/train_all.scp ]; then
    mv $fbankdir/train_all.scp $fbankdir/train_all.scp.bak
  fi

  for train in train_clean_100; do
    cat data/$train/feats.scp >> $fbankdir/train_all.scp
  done
  compute-cmvn-stats --binary=false scp:$fbankdir/train_all.scp $fbankdir/train_global_cmvn.txt
fi

if [ $stage -le 3 ]; then
  # train share encoder with apc loss function
  exp_dir=exp/tsfm_shrenc_l0r3s4_lr4e-4_anneal15_08_ep30
  if [ ! -d $exp_dir ]; then
    mkdir -p $exp_dir
  fi

  CUDA_VISIBLE_DEVICES='0' bin/train_apc.py --exp_dir $exp_dir \
    --train_config config/share_encoder.yaml \
    --data_config config/data.yaml \
    --learning_rate 0.0004 \
    --batch_size 32 \
    --epochs 30 \
    --time_shift 1 \
    --load_data_workers 2 \
    --anneal_lr_epoch 15 \
    --anneal_lr_ratio 0.8 \
    --print_freq 100 > $exp_dir/train.log 2>&1 &
fi

if [ $stage -le 4 ]; then
  # train a phoneme classifier with CTC loss function
  exp_dir=exp/tsfm_shrenc_l0r3s4_lr4e-4_anneal15_08_ep30
  if [ ! -d $exp_dir ]; then
    mkdir -p $exp_dir
  fi

  CUDA_VISIBLE_DEVICES='0' bin/train_apc.py --exp_dir $exp_dir \
    --train_config config/share_encoder.yaml \
    --data_config config/data.yaml \
    --learning_rate 0.0004 \
    --batch_size 32 \
    --epochs 30 \
    --time_shift 1 \
    --load_data_workers 2 \
    --anneal_lr_epoch 15 \
    --anneal_lr_ratio 0.8 \
    --print_freq 100 > $exp_dir/train.log 2>&1 &
fi





