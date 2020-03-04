
import copy
import torch.nn as nn
import editdistance as ed
from models.layers import *
from models.rnn_apc import Generator

class CTCNet(nn.Module):
    def __init__(self, shared_encoder, input_layer, lstm_layer, N_ctc, predict_layer, dropout):
        super(CTCNet, self).__init__()
        self.shared_encoder = shared_encoder
        self.input_layer = input_layer
        self.ctc_layers = clones(lstm_layer, N_ctc-1)
        self.dropout = nn.Dropout(p=dropout)
        self.predictor = predict_layer
        
    def forward(self, x):
        x = self.shared_encoder(x, activation=True)
        x = x.contiguous().view(x.size(0),int(x.size(1)/2),x.size(2)*2)
        x, _ = self.input_layer(x)
        for layer in self.ctc_layers:
            y, _ = layer(x)
            x = x + self.dropout(y)
        return x, self.predictor(x)

class ASRDecoder(nn.Module):
    def __init__(self, ctc_net, d_model, d_te, out_size, dropout):
        super(ASRDecoder, self).__init__()
        self.ctc_net = ctc_net
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.pooling = StatsPoolingLayer()
        self.fc = nn.Sequential(nn.Linear(2 * d_model, d_model),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_model, d_te))
        self.predictor = Generator(d_te, out_size)
        
    def forward(self, x, x_sizes):
        x, ctc_out = self.ctc_net(x)
        ctc_out = self.log_softmax(ctc_out)
        x = self.pooling(x, x_sizes)
        t_embedding = self.fc(x)
        out = self.predictor(t_embedding)
        return ctc_out, out, t_embedding

    def compute_wer(self, index, input_sizes, targets, target_sizes):
        batch_errs = 0
        batch_tokens = 0
        for i in range(len(index)):
            label = targets[i][:target_sizes[i]]
            pred = []
            for j in range(len(index[i][:input_sizes[i]])):
                if index[i][j] == 0:
                    continue
                if j == 0:
                    pred.append(index[i][j])
                if j > 0 and index[i][j] != index[i][j-1]:
                    pred.append(index[i][j])
            batch_errs += ed.eval(label, pred)
            batch_tokens += len(label)
        return batch_errs, batch_tokens


def make_model(shared_encoder, args):
    num_direction = 2 if args.bidirection else 1
    input_layer = nn.LSTM(input_size=args.d_model*2, hidden_size=args.d_hidden, batch_first=True, bidirectional=args.bidirection)
    lstm_layer = nn.LSTM(input_size=num_direction * args.d_hidden, hidden_size=args.d_hidden, batch_first=True, bidirectional=args.bidirection)
    predict_layer = Generator(num_direction * args.d_hidden, args.phone_size + 1)
    ctc_net = CTCNet(shared_encoder, input_layer, lstm_layer, args.N_ctc, predict_layer, args.dropout)
    model = ASRDecoder(ctc_net, num_direction * args.d_hidden, args.d_te, args.phrase_size, args.dropout)
   
    # this initialization is really important to make the training converge 
    for name, p in model.named_parameters():
        if p.dim() > 1 and not name.startswith('ctc_net.shared_encoder'):
            nn.init.xavier_uniform_(p)
    return model


