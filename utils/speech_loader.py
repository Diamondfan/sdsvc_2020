#encoding=utf-8

import torch
import kaldiio
from torch.utils.data import Dataset, DataLoader
from utils.feat_op import *

class SpeechDataset(Dataset):
    def __init__(self, scp_path, phrase_lab_path=None, spk_lab_path=None, left_context=0, right_context=0, skip_frame=1):
        self.left_context = left_context
        self.right_context = right_context
        self.skip_frame = skip_frame        
        self.load_data(scp_path, phrase_lab_path, spk_lab_path)

    def load_data(self, scp_path, phrase_lab_path=None, spk_lab_path=None):
        ark_dict = self.load_feature(scp_path)
        
        if phrase_lab_path != None:
            phrase_dict = self.load_label(phrase_lab_path)
            assert (len(ark_dict) - len(phrase_dict)) < 5 
        
        if spk_lab_path != None:
            spk_dict = self.load_label(spk_lab_path)
            assert len(ark_dict) - len(spk_dict) < 5
        
        self.items = []
        for i in range(len(ark_dict)):
            utt, ark_path = ark_dict[i]
            phrase, spk = None, None
            if phrase_lab_path != None:
                phrase = phrase_dict[utt]
            if phrase_lab_path != None:
                spk = spk_dict[utt]
            self.items.append((utt, ark_path, phrase, spk))

    def load_feature(self, scp_path):
        #read the ark path
        ark_dict = []
        with open(scp_path, 'r') as fin:
            line = fin.readline()
            while line:
                utt, path = line.strip().split(' ')
                ark_dict.append((utt, path))
                line = fin.readline()
        print("Reading %d lines from %s" % (len(ark_dict), scp_path))
        return ark_dict
    
    def load_label(self, lab_path):
        #read the label
        label_dict = dict()
        with open(lab_path, 'r') as fin:
            line = fin.readline()
            while line:
                utt, label = line.strip().split(' ', 1)
                label_dict[utt] = label
                line = fin.readline()
        print("Reading %d lines from %s" % (len(label_dict), lab_path))
        return label_dict
        
    def __getitem__(self, idx):
        utt, ark_path, phrase, spk = self.items[idx]
        feat = kaldiio.load_mat(ark_path)
        seq_len, dim = feat.shape
        if seq_len % self.skip_frame != 0:
            pad_len = self.skip_frame - seq_len % self.skip_frame
            feat = np.vstack([feat,np.zeros((pad_len, dim))])
        feat = skip_feat(context_feat(feat, self.left_context, self.right_context), self.skip_frame)
        return (utt, feat, phrase, spk)

    def __len__(self):
        return len(self.items)

class SpeechDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, num_workers=0, timeout=1000):
        super(SpeechDataLoader, self).__init__(dataset, 
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                collate_fn=self.collate_fn,
                                                drop_last=False,
                                                timeout=timeout)

    def collate_fn(self, batch):
        #Todo: adapt with speaker label
        inputs_max_length = max(x[1].size(0) for x in batch)
        feat_size = batch[0][0].size(1)
        targets_max_length = max(x[2].size(0) for x in batch)
        batch_size = len(batch)
        
        inputs = torch.zeros(batch_size, inputs_max_length, feat_size)
        targets = torch.zeros(batch_size, targets_max_length)
        utt_list = []

        for x in range(batch_size):
            feature, label, utt = batch[x]
            feature_length = feature.size(0)
            label_length = label.size(0)
            
            inputs[x].narrow(0, 0, feature_length).copy_(feature)
            targets[x].narrow(0, 0, label_length).copy_(label)
            utt_list.append(utt)
        return inputs.float(), targets.long(), utt_list


if __name__ == "__main__":
    train_label = 'data/train/phn_text'
    train_scp = 'data/train/fbank.scp'
    train_dataset = SpeechDataset(train_scp, train_label, '', skip_frame=4)
    #train_loader = SpeechDataLoader(train_dataset, batch_size=10, shuffle=False, num_workers=4)

