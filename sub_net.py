from keras import optimizers
from gv_tools.util.logger import Logger
from keras.models import Sequential
import os
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout, GRU, Bidirectional, Embedding, Merge
from keras.layers.merge import Add, Concatenate
from keras.layers.core import RepeatVector
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.callbacks import History


class SubNet:
    def __init__(self, logger: Logger, sub_name: str, pred_len=20, embedding_size=128, hidden_size=128,
                 obs_len=20, batch_size=128):
        self.net = None
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self._name = sub_name
        self.logger = logger

    def build_network(self, learning_rate):
        self.logger.field("Start Building Network", self._name)
        self.net = Sequential()
        self.net.add(Dense(self.embedding_size, activation='relu', input_shape=(self.obs_len, 2)))
        self.net.add(GRU(self.hidden_size,
                         activation='relu',
                         input_shape=(self.obs_len, 2),
                         batch_size=self.batch_size,
                         return_sequences=True,
                         stateful=False,
                         dropout=0.0))
        self.net.add(GRU(self.hidden_size,
                         activation='relu',
                         input_shape=(self.obs_len, 2),
                         batch_size=self.batch_size,
                         return_sequences=False,
                         stateful=False,
                         dropout=0.0))
        self.net.add(RepeatVector(self.pred_len))
        self.net.add(GRU(self.hidden_size,
                         activation='relu',
                         input_shape=(self.pred_len, 2),
                         batch_size=self.batch_size,
                         return_sequences=True,
                         stateful=False,
                         dropout=0.0))
        self.net.add(GRU(self.hidden_size,
                         activation='relu',
                         input_shape=(self.pred_len, 2),
                         batch_size=self.batch_size,
                         return_sequences=True,
                         stateful=False,
                         dropout=0.0))
        self.net.add(TimeDistributed(Dense(2)))
        self.net.add(Activation('linear'))
        opt = optimizers.RMSprop(lr=learning_rate)
        self.net.compile(optimizer=opt,
                         loss='mse')
        # self.net.summary()

    def build_bi_subnet(self, learning_rate):
        left = Sequential()
        left.add(Dense(self.embedding_size, activation='relu', input_shape=(self.obs_len, 2)))

        left.add(
            LSTM(self.hidden_size, input_shape=(self.obs_len, 2), batch_size=self.batch_size, return_sequences=False))
        right = Sequential()
        right.add(Dense(self.embedding_size, activation='relu', input_shape=(self.obs_len, 2)))

        right.add(
            LSTM(self.hidden_size, input_shape=(self.obs_len, 2), batch_size=self.batch_size, return_sequences=False,
                 go_backwards=True))

        self.net = Sequential()
        self.net.add(Merge([left, right], mode='sum'))
        self.net.add(RepeatVector(self.pred_len))
        self.net.add(GRU(self.hidden_size,
                         activation='relu',
                         input_shape=(self.pred_len, 2),
                         batch_size=self.batch_size,
                         return_sequences=True,
                         stateful=False,
                         dropout=0.0))
        self.net.add(GRU(self.hidden_size,
                         activation='relu',
                         input_shape=(self.pred_len, 2),
                         batch_size=self.batch_size,
                         return_sequences=True,
                         stateful=False,
                         dropout=0.0))
        self.net.add(TimeDistributed(Dense(2)))
        self.net.add(Activation('linear'))
        opt = optimizers.RMSprop(lr=learning_rate)
        self.net.compile(optimizer=opt,
                         loss='mse')


    def train(self, x, y, epoch):
        self.logger.log('Start Training ...')
        self.logger.field('total training epochs', epoch)

        self.net.fit(x, y,
                     batch_size=self.batch_size,
                     epochs=epoch,
                     validation_split=0.1,
                     verbose=1)

    def save_network(self, save_path, save_name):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_path = save_path + '/' + save_name + self._name
        self.net.save(save_path + '.h5')

    def predict(self, x):
        if len(np.shape(x)) == 2:
            x = np.reshape(x, [-1, self.obs_len, 2])

        if len(np.shape(x)) == 3:
            predicted = self.net.predict(x, batch_size=self.batch_size)
            return predicted
        else:
            self.logger.field('Wrong Input Data Shape', np.shape(x))
            self.logger.error('Wrong Input Data Shape for predicting')
            return None

    def bi_predict(self, x):
        if len(np.shape(x)) == 2:
            x = np.reshape(x, [-1, self.obs_len, 2])

        if len(np.shape(x)) == 3:
            predicted = self.net.predict([x, x], batch_size=self.batch_size)
            return predicted
        else:
            self.logger.field('Wrong Input Data Shape', np.shape(x))
            self.logger.error('Wrong Input Data Shape for predicting')
            return None


    @property
    def sub_net_name(self):
        return self._name



