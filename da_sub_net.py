import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


def make_mlp(dim_list, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))

        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class VanillaNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim=2, obs_len=20, pred_len=20, drop_out=0.5, gpu=False,
                 oracle_init=True, da_transform=False):
        super(VanillaNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.obs_len = obs_len
        self.output_dim = output_dim
        self.gpu = gpu
        self.pred_len = pred_len

        self.gru_layers_num = 1
        self.da_transform_flag = da_transform
        self.da_transform = make_mlp([2*obs_len, 5*obs_len, 2*obs_len])

        self.loc_embedding = nn.Linear(output_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, dropout=drop_out, num_layers=self.gru_layers_num)
        self.gru_decode = nn.GRU(embedding_dim, hidden_dim, batch_first=True, dropout=drop_out, num_layers=self.gru_layers_num)
        self.gru2out = nn.Linear(hidden_dim, output_dim)

        if oracle_init:
            for p in self.parameters():
                nn.init.normal(p, 0, 1)

    def init_hidden(self, batch_size=1):
        h = Variable(torch.zeros(self.gru_layers_num, batch_size, self.hidden_dim))
        if self.gpu:
            return h.cuda()
        else:
            return h

    def mlp(self, out):
        out = self.gru2out(out)
        return out

    def forward(self, obs_trajectory):

        hidden = self.init_hidden(batch_size=obs_trajectory.size(0))

        if self.da_transform_flag:
            obs_trajectory = obs_trajectory.view(-1, self.obs_len * 2)
            obs_trajectory = self.da_transform(obs_trajectory)
            obs_trajectory = obs_trajectory.view(-1, self.obs_len, 2)

        pred = []
        for i, input_t in enumerate(obs_trajectory.chunk(obs_trajectory.size(1), dim=1)):
            emb = self.loc_embedding(input_t)
            self.gru.flatten_parameters()
            out, hidden = self.gru(emb, hidden)
        last_time_pred = self.mlp(out)
        for j in range(self.pred_len):
            emb = self.loc_embedding(last_time_pred)
            out, hidden = self.gru_decode(emb, hidden)
            last_time_pred = self.mlp(out)
            pred.append(torch.squeeze(last_time_pred, dim=1))

        return torch.stack(pred, dim=1)



