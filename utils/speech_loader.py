#encoding=utf-8

import torch
import kaldiio
from torch.utils.data import Dataset, DataLoader
from feat_op import *

class SingleSet(object):
    def __init__(self, data_path):
        self.name = data_path['type']
        scp_path = data_path['scp_path']
        ark_dict = self._load_feature(scp_path)
        
        if 'text_label' in data_path:
            text_dict = self._load_label(data_path['text_label'])
            assert (len(ark_dict)-len(text_dict))<5, "label and sample size mismatch"

        if 'phrase_label' in data_path:
            phrase_dict = self._load_label(data_path['phrase_label'])
            assert (len(ark_dict)-len(phrase_dict))<5, "label and sample size mismatch"
        
        if 'speaker_label' in data_path:
            spk_dict = self._load_label(data_path['speaker_label'])
            assert (len(ark_dict)-len(spk_dict))<5, "label and sample size mismatch"
        
        self.items = []
        for i in range(len(ark_dict)):
            utt, ark_path = ark_dict[i]
            if 'text_label' in data_path:
                text = text_dict[utt]
            else:
                text = [1]
            if 'phrase_label' in data_path:
                phrase = phrase_dict[utt]
            else:
                phrase = [1]
            if 'speaker_label' in data_path:
                speaker = spk_dict[utt]
            else:
                speaker = [1]
            self.items.append((utt, ark_path, text, phrase, speaker))
        
    def get_len(self):
        return len(self.items)

    def _load_feature(self, scp_path):
        ark_dict = []
        with open(scp_path, 'r') as fin:
            line = fin.readline()
            while line:
                utt, path = line.strip().split(' ')
                ark_dict.append((utt, path))
                line = fin.readline()
        print("Reading %d lines from %s" % (len(ark_dict), scp_path))
        return ark_dict
    
    def _load_label(self, lab_path):
        label_dict = dict()
        with open(lab_path, 'r') as fin:
            line = fin.readline()
            while line:
                utt, label = line.strip().split(' ', 1)
                label_dict[utt] = [int(j) for j in label.split(' ')]
                line = fin.readline()
        print("Reading %d lines from %s" % (len(label_dict), lab_path))
        return label_dict

class SpeechDataset(Dataset):
    def __init__(self, data_paths, left_context=0, right_context=0, skip_frame=1):
        self.left_context = left_context
        self.right_context = right_context
        self.skip_frame = skip_frame     
        self.data_streams = self._load_streams(data_paths)
        self.data_stream_sizes = [i.get_len() for i in self.data_streams]
        self.data_stream_cum_sizes = [self.data_stream_sizes[0]]
        for i in range(1, len(self.data_stream_sizes)):
            self.data_stream_cum_sizes.append(self.data_stream_cum_sizes[-1] + self.data_stream_sizes[i])

    def _load_streams(self, data_paths):
        data_streams = []
        for i in data_paths:
            stream = SingleSet(data_paths[i])
            data_streams.append(stream)
        return data_streams
                    
    def __getitem__(self, idx):
        stream_idx = -1
        for i in range(len(self.data_stream_cum_sizes)):
            if idx < self.data_stream_cum_sizes[i]:
                stream_idx = i
                break
        if stream_idx == -1:
            raise Exception('index exceed.')
        if stream_idx == 0:
            internal_idx = idx
        else:
            internal_idx = idx - self.data_stream_cum_sizes[stream_idx-1]
        
        utt, ark_path, text, phrase, spk = self.data_streams[stream_idx].items[internal_idx]
        feat = kaldiio.load_mat(ark_path)
        seq_len, dim = feat.shape
        if seq_len % self.skip_frame != 0:
            pad_len = self.skip_frame - seq_len % self.skip_frame
            feat = np.vstack([feat,np.zeros((pad_len, dim))])
        feat = skip_feat(context_feat(feat, self.left_context, self.right_context), self.skip_frame)
        return (utt, feat, text, phrase, spk)

    def __len__(self):
        return sum(self.data_stream_sizes)

class SpeechDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, num_workers=0, timeout=1000):
        super(SpeechDataLoader, self).__init__(dataset, 
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                collate_fn=self.collate_fn,
                                                drop_last=False)
                                                #timeout=timeout)

    def collate_fn(self, batch):
        feats_max_length = max(x[1].shape[0] for x in batch)
        feat_size = batch[0][1].shape[1]
        text_max_length = max(len(x[2]) for x in batch)
        batch_size = len(batch)
        
        feats = torch.zeros(batch_size, feats_max_length, feat_size)
        texts = torch.zeros(batch_size, text_max_length)
        utt_list = []
        phrases = []
        speakers = []

        for x in range(batch_size):
            utt, feature, text, phrase, speaker = batch[x]
            feature_length = feature.shape[0]
            text_length = len(text)
            
            feats[x].narrow(0, 0, feature_length).copy_(torch.Tensor(feature))
            texts[x].narrow(0, 0, text_length).copy_(torch.Tensor(text))
            utt_list.append(utt)
            phrases.append(phrase)
            speakers.append(speaker)
        return utt_list, feats.float(), texts.long(), torch.LongTensor(phrases), torch.LongTensor(speakers)

if __name__ == "__main__":
    import yaml
    data_paths = yaml.safe_load(open('config/data.yaml', 'r'))
    train_dataset = SpeechDataset(data_paths['train_data_path'])
    train_loader = SpeechDataLoader(train_dataset, batch_size=10, shuffle=False)
    for i, data in enumerate(train_loader):
        utt_list, feats, text, phrase, speaker = data
        print(speaker.size())
        break


