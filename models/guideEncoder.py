import torch.nn as nn
import torch.nn.functional as F
from .SimAttConv import SimAttConv


class guideEncoder(nn.Module):
    def __init__(self,
                 num_layers,
                 nfeat,
                 nhid,
                 nclass,
                 heads,
                 negative_slope,
                 concat,
                 dropout):
        super(guideEncoder, self).__init__()
        self.body = guideEncoder_body(num_layers, nfeat, nhid, heads, negative_slope, concat, dropout)
        self.fc = nn.Linear(nhid, nclass)
        self.activation = nn.ReLU()
        self.bn = nn.LayerNorm(nhid)
    def forward(self, x, edge_index, edge_weight, return_attention_weights=None):
        if isinstance(return_attention_weights, bool) and return_attention_weights==True:
            logits, attention_weights = self.body(x, edge_index, edge_weight, return_attention_weights=return_attention_weights)
            logits = self.activation(self.bn(logits))
            logits = self.fc(logits)
            return logits, attention_weights
        else:
            logits = self.body(x, edge_index, edge_weight)
            logits = self.activation(self.bn(logits))
            logits = self.fc(logits)
            return logits



class guideEncoder_body(nn.Module):
    def __init__(self,
                 num_layers,
                 nfeat,
                 nhid,
                 heads,
                 negative_slope,
                 concat,
                 dropout):
        super(guideEncoder_body, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        self.layers.append(SimAttConv(
            nfeat, nhid, heads, self.concat, self.negative_slope, self.dropout))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the nfeat = nhid * num_heads
            if self.concat == True:
                self.layers.append(SimAttConv(
                nhid * heads, nhid, heads, self.concat, self.negative_slope))
            else:
                self.layers.append(SimAttConv(
                nhid, nhid, heads, self.concat, self.negative_slope))

        
    def forward(self, x, edge_index, edge_weight, return_attention_weights=None):
        h = x
        for l in range(self.num_layers-1):
            h = self.layers[l](h, edge_index, edge_weight).flatten(1)

        if isinstance(return_attention_weights, bool) and return_attention_weights==True:
            logits, attention_weights = self.layers[-1](h, edge_index, edge_weight, return_attention_weights=return_attention_weights)
            return logits, attention_weights
        else:
            logits = self.layers[-1](h, edge_index, edge_weight)  # .mean(1)
            return logits



