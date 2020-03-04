#!/bin/bash

# 

libri_path=/home/vijaysumaravi/Documents/database/librispeech/
deepmine_path=/home/vijaysumaravi/Documents/database/sdsvc_2020/data/task1/
stage=4

. ./cmd.sh
. ./path.sh
. parse_options.sh

# Step1. Use librispeech, voxceleb and Deepmine to do unsupervied pretraining.
# Test the code woth only train clean 100h first
if [ $stage -le 0 ]; then
  #format librispeech data as Kaldi data directories
  for part in dev-clean train-clean-100; do
    local/libri_data_prep.sh $libri_path/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done
  #todo: scripts needed to process voxceleb and deepmine dataset
  local/deepmine_data_prep.sh $deepmine_path
fi

if [ $stage -le 1 ]; then
  # Prepare label for ASR part, including phoneme sequence and phrase id
  # We need the label file to be int
  lexicon=resources/librispeech-lexicon.txt
  phone=resources/phones.txt
  awk '{for (i=2; i<=NF; ++i) { gsub(/[0-9]/, "", $i); print $i}}' $lexicon | sort | uniq > $phone 
  
  for part in dev_clean train_clean_100; do
    local/word_to_phoneme.py $lexicon data/$part/text data/$part/text_phone
    local/phone2id.py $phone data/$part/text_phone data/$part/text_phone_id
  done
    
  for part in train_deepmine dev_deepmine; do
    local/phone2id.py $phone data/$part/text_phone data/$part/text_phone_id
  done
fi

if [ $stage -le 2 ]; then
  #feature extraction for librispeech, filter-bank
  fbankdir=data/fbank
  
  for part in dev_clean train_clean_100; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 10 data/$part exp/make_fbank/$part $fbankdir  
  done
  
  for part in train_deepmine dev_deepmine test_deepmine; do
    local/make_fbank.sh --cmd "$train_cmd" --nj 10 data/$part exp/make_fbank/$part $fbankdir  
  done

  # put training scp to on scp to caculate the cmvn
  if [ -f $fbankdir/train_all.scp ]; then
    mv $fbankdir/train_all.scp $fbankdir/train_all.scp.bak
  fi

  for train in train_clean_100 train_deepmine; do
    cat data/$train/feats.scp >> $fbankdir/train_all.scp
  done
  compute-cmvn-stats --binary=false scp:$fbankdir/train_all.scp $fbankdir/train_global_cmvn.txt
fi

if [ $stage -le 3 ] && false; then
  # train share encoder with apc loss function
  exp_dir=exp/shrenc_rnn_noskip_shift2_lr4e-4_anneal15_08_ep30/
  if [ ! -d $exp_dir ]; then
    mkdir -p $exp_dir
  fi

  CUDA_VISIBLE_DEVICES='0' bin/train_apc.py --exp_dir $exp_dir \
    --train_config config/share_encoder.yaml \
    --data_config config/data.yaml \
    --use_cmvn \
    --net_type 'rnn' \
    --learning_rate 0.0004 \
    --batch_size 32 \
    --epochs 15 \
    --time_shift 2 \
    --resume_encoder 'exp/shrenc_rnn_noskip_shift2_lr4e-4_anneal15_08_ep30/model.6.mdl' \
    --start_epoch 7 \
    --load_data_workers 2 \
    --anneal_lr_epoch 8 \
    --anneal_lr_ratio 0.8 \
    --print_freq 100 >> $exp_dir/train.log 2>&1 &
fi

if [ $stage -le 4 ] && true; then
  # train a phoneme classifier with CTC loss function
  exp_dir=exp/randomenc_update_rnnctc_pid02_lr2e-4_anneal7_ds2/
  shared_encoder_model='' #'exp/shrenc_rnn_noskip_shift2_lr4e-4_anneal15_08_ep30/best.mdl'
  if [ ! -d $exp_dir ]; then
    mkdir -p $exp_dir
  fi

  CUDA_VISIBLE_DEVICES='0' bin/train_asr.py --exp_dir $exp_dir \
    --train_config config/asr.yaml \
    --data_config config/data_asr.yaml \
    --learning_rate 0.0002 \
    --batch_size 16 \
    --epochs 12 \
    --load_data_workers 2 \
    --anneal_lr_epoch 7 \
    --anneal_lr_ratio 0.5 \
    --phrase_lambda 0.2 \
    --use_cmvn \
    --print_freq 100 > $exp_dir/train.log 2>&1 &
    #--resume_encoder $shared_encoder_model \
    #--fix_encoder \
fi

if [ $stage -le 5 ] && false; then
  # generate a logprob and embedding for deepmine test set.
  load_model='exp/rnnenc_fixenc_rnnctc_pid02_lr2e-4_anneal7_ds2/best.mdl'
  out_logprob='data/test_deepmine/utt2logprob' 
  out_embedding='data/test_deepmine/tembedding'
  
  CUDA_VISIBLE_DEVICES='0' bin/test_deepmine.py \
    --test_config config/test_deepmine.yaml \
    --batch_size 16 \
    --load_data_workers 2 \
    --resume_model $load_model \
    --use_cmvn \
    --out_prob $out_logprob \
    --out_embedding $out_embedding \
    --print_freq 100

  trials=/home/vijaysumaravi/Documents/database/sdsvc_2020/data/task1/docs/trials.txt
  model_enrollment=/home/vijaysumaravi/Documents/database/sdsvc_2020/data/task1/docs/model_enrollment.txt
  ivector_scores=upload/team25_sub1/answer.txt
  new_scores=upload/team25_sub1/text_prob.txt
  local/get_new_score.py $out_logprob $trials $model_enrollment $ivector_scores $new_scores
fi





