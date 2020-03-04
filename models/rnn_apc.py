
import copy
import torch.nn as nn
from models.layers import *

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)

class RNNSharedEncoder(nn.Module):
    '''Core encoder is a stack of N layers, we use uni-direction structure
    which is controlled by mask'''
    def __init__(self, enc_layer, attention_layer, N, predict_layer, d_model, dropout):
        super(RNNSharedEncoder, self).__init__()
        self.layers = clones(attention_layer, N)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(d_model)
        self.embedding = enc_layer
        self.predictor = predict_layer
        
    def forward(self, x, activation=False):
        x = self.embedding(x)
        for layer in self.layers:
            y, _ = layer(self.norm(x))
            x = x + self.dropout(y)
        if activation:
            return x
        else:
            return self.predictor(x)

def make_model(input_size, N=6, N_embed=2, d_model=512, d_ff=2048, h=8, dropout=0.1):
    model = RNNSharedEncoder(
        FeatureEmbedding(input_size, N_embed, d_model, dropout),
        nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True), N,
        Generator(d_model, input_size), d_model, dropout)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


