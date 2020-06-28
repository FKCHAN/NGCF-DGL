


import torch.nn as nn
from NGCF.NGCF_Conv import NGCF_Conv

class NGCF_GCN(nn.Module):
    def __init__(self,
                 g,
                 dim_emb,
                 n_layers,
                 activation,
                 dropout):
        super(NGCF_GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(NGCF_Conv(dim_emb, dim_emb, activation=activation))
        # output layer
        self.layers.append(NGCF_Conv(dim_emb, dim_emb))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h