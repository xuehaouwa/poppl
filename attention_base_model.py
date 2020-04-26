import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class Attention(nn.Module):
    """
    use batch_first = False mode
    """

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        self.v.data.normal_(mean=0, std=1. / np.sqrt(self.v.size(0)))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        h = hidden[-1].repeat(max_len, 1, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        attn_energies = self.score(h, encoder_outputs)  
        return F.softmax(attn_energies, dim=1)  

    def score(self, hidden, encoder_outputs):

        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.permute(1, 2, 0)  
        v = self.v.repeat(encoder_outputs.size(1), 1).unsqueeze(1) 
        energy = torch.bmm(v, energy)

        return energy.squeeze(1)


class Attention_general(nn.Module):
    """
    use batch_first = False mode
    """

    def __init__(self, hidden_dim):
        super(Attention_general, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim, hidden_dim)

    def forward(self, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        attn_energies = self.score(hidden, encoder_outputs)  
        return F.softmax(attn_energies, dim=1)  

    def score(self, hidden, encoder_outputs):

        energy = F.tanh(self.attn(encoder_outputs))
        energy = energy.permute(1, 2, 0) 
        hidden = hidden.permute(1, 0, 2)
        energy = torch.bmm(hidden, energy)  

        return energy.squeeze(1) 


class ASubNet(nn.Module):
    USE_GPU = False

    def __init__(self, hidden_dim, embedding_dim, dropout, pred_len, general=False):
        super(ASubNet, self).__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.loc_embedding = nn.Linear(2, embedding_dim)
        self.loc_embedding_dec = nn.Linear(2, embedding_dim)
        self.gru_loc = nn.GRU(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.gru_loc_dec = nn.GRU(embedding_dim + hidden_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.tanh = nn.Tanh()
        if general:
            self.temporal_attention_loc = Attention_general(hidden_dim=hidden_dim)
        else:
            self.temporal_attention_loc = Attention(hidden_dim=hidden_dim)
        self.out2loc = nn.Linear(hidden_dim, 2)

    def predict_one_step(self, last_loc, enc_out_loc, hidden_loc):
        loc_embedded = self.loc_embedding(last_loc)
        loc_embedded = self.tanh(loc_embedded)
        temporal_weight_loc = self.temporal_attention_loc(hidden_loc, enc_out_loc)
        context_loc = temporal_weight_loc.unsqueeze(1).bmm(enc_out_loc)
        emb_con_loc = torch.cat((loc_embedded, context_loc), dim=2)

        dec_out_loc, hidden_loc = self.gru_loc_dec(emb_con_loc, hidden_loc)
        self.gru_loc_dec.flatten_parameters()

        out_loc = self.out2loc(dec_out_loc)

        return hidden_loc, out_loc

    def forward(self, obs):
        hidden_loc = self.init_hidden(batch_size=obs.size(0))
        predicted = []
        # encoding
        obs_loc = torch.index_select(obs, dim=2, index=self.generate_index([0, 1], use_gpu=self.USE_GPU))
        loc_encoder_outputs = []

        for i, input_t_loc in enumerate(obs_loc.chunk(obs_loc.size(1), dim=1)):
            emb_loc = self.loc_embedding(input_t_loc)
            emb_loc = self.tanh(emb_loc)
            self.gru_loc.flatten_parameters()
            enc_out_loc, hidden_loc = self.gru_loc(emb_loc, hidden_loc)
            loc_encoder_outputs.append(enc_out_loc)

        out_loc = self.out2loc(enc_out_loc)
        loc_encoder_outputs = torch.cat(loc_encoder_outputs, dim=1)

        for _ in range(self.pred_len):
            hidden_loc, out_loc = self.predict_one_step(out_loc, loc_encoder_outputs, hidden_loc)
            predicted.append(torch.squeeze(out_loc, dim=1))

        return torch.stack(predicted, dim=1)

    def init_hidden(self, batch_size=1):
        h = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        if self.USE_GPU:
            return h.cuda()
        else:
            return h

    @staticmethod
    def generate_index(index, use_gpu=True):
        if use_gpu:
            return Variable(torch.LongTensor(index)).cuda()
        else:
            return Variable(torch.LongTensor(index))
