
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

class SharedEncoder(nn.Module):
    '''Core encoder is a stack of N layers, we use uni-direction structure
    which is controlled by mask'''
    def __init__(self, enc_layer, attention_layer, N, predict_layer):
        super(SharedEncoder, self).__init__()
        self.layers = clones(attention_layer, N)
        self.norm = LayerNorm(attention_layer.size)
        self.embedding = enc_layer
        self.predictor = predict_layer
        
    def forward(self, x, mask, activation=False):
        "Pass the input (and mask) through each layer in turn."
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        if activation:
            return x
        else:
            return self.predictor(x)

def make_model(input_size, N=6, N_embed=2, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = SharedEncoder(
        nn.Sequential(FeatureEmbedding(input_size, N_embed, d_model, dropout), c(position)),
        EncoderLayer(d_model, c(attn), c(ff), dropout), N,
        Generator(d_model, input_size))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


