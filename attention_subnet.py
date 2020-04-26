from gv_tools.util.logger import Logger
from attention_base_model import ASubNet
import torch
from data.dataloader import TrajectoryDataLoader
import torch.utils.data as Data
import os
import numpy as np
from torch.autograd import Variable
import torch.nn as nn


class AttSub:
    USE_GPU = False
    LOSS_FUNC = nn.MSELoss()

    def __init__(self, logger: Logger, sub_name: str, pred_len=20, embedding_size=128, hidden_size=128,
                 obs_len=20, batch_size=12, dropout=0.1):
        self.net = None
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = dropout
        self.batch_size = batch_size
        self.name = sub_name
        self.logger = logger
        self.optimizer = None
        self.train_obs = None
        self.train_pred = None
        self.dataloader = None



    def build_dataloader(self, train_obs, train_pred):
        self.logger.field('Start Building Data Loader', self.name)
        self.train_obs = train_obs
        self.train_pred = train_pred
        dataset = TrajectoryDataLoader(self.train_obs, self.train_pred)
        self.dataloader = Data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1
        )
        self.logger.log('Train Data Loader Finished!')

    def build_network(self, lr, general=False):
        self.logger.field("Start Build Att SubNet", self.name)
        self.net = ASubNet(hidden_dim=self.hidden_size, embedding_dim=self.embedding_size, dropout=self.drop,
                           pred_len=self.pred_len, general=general)
        if self.USE_GPU:
            self.net.cuda()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def train(self, epochs, verbose_step=50):

        epoch = 0
        self.net.train()

        while epoch < epochs:
            epoch += 1
            losses = []
            for i, (batch_x, batch_y) in enumerate(self.dataloader):
                if self.USE_GPU:
                    data, target = Variable(batch_x).cuda(), Variable(batch_y).cuda()
                else:
                    data, target = Variable(batch_x), Variable(batch_y)
                self.optimizer.zero_grad()
                out = self.net(data)
                loss = self.LOSS_FUNC(out, target)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.data.cpu()[0])
            if epoch % verbose_step == 0:
                self.logger.field('Epoch', epoch)
                self.logger.field('loss', np.mean(losses))

    def save_model(self, save_path, save_name):
        save_path = save_path + '/' + save_name + self.name
        torch.save(self.net.state_dict(), save_path + '_params.pkl')
        torch.save(self.net, save_path + '.pkl')

    def evaluate(self):
        pass

    def predict_one(self, obs):
        if self.USE_GPU:
            obs = Variable(torch.Tensor(np.expand_dims(obs, axis=0))).cuda()
        else:
            obs = Variable(torch.Tensor(np.expand_dims(obs, axis=0)))
        pred = self.net(obs)
        predicted_one = pred.data.cpu().numpy()
        predicted_one = np.reshape(predicted_one, [1, self.pred_len, 2])

        return predicted_one









