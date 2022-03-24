import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import fvcore.nn.weight_init as weight_init
from detectron2.layers import Linear, ShapeSpec


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, 
        dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        if self.num_layers > 0:
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(
                nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
            if self.dropout > 0.0:
                self.dropouts = nn.ModuleList(
                    nn.Dropout(dropout) for _ in range(self.num_layers - 1))
        else:
            self.layers = []

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if i < self.num_layers - 1 and self.dropout > 0.0:
                x = self.dropouts[i](x)
        return x

class ATTWeightHead(nn.Module):
    def __init__(self, feature_dim, num_layers, dropout):
        super().__init__()
        self.weight_out_dim = feature_dim
        self.q_proj = MLP(
            feature_dim, self.weight_out_dim, self.weight_out_dim, 
            num_layers, dropout)
        self.k_proj = MLP(
            feature_dim, feature_dim, self.weight_out_dim, 
            num_layers, dropout)


    def forward(self, query, key, temp_embs=None):
        '''
        Inputs:
          query: B x M x F
          key: B x N x F
          temp_embs: B x N x F
        '''
        k = self.k_proj(key) # B x N x D
        q = self.q_proj(query) # B x M x D
        attn_weights = torch.bmm(q, k.transpose(1, 2)) # B x M x N
        return attn_weights


class FCHead(nn.Module):
    """
    """

    def __init__(self, input_shape, fc_dim, num_fc):
        """
        """
        super().__init__()
        fc_dims = [fc_dim for _ in range(num_fc)]
        assert len(fc_dims) > 0
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        self.fcs = []
        for k, x in enumerate(fc_dims):
            fc = Linear(np.prod(self._output_size), x)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = x

        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_shape(self):
        """
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])