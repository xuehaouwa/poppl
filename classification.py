from keras import optimizers
from gv_tools.util.logger import Logger
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout, GRU, Bidirectional, Embedding, Merge
import os
from keras.models import load_model
from keras.optimizers import SGD
import heapq
import numpy as np
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization


class ClassificationNet:
    KERNEL_SIZE = 3
    OUT_CHANNEL = 64
    POOL_SIZE = 2

    def __init__(self, logger: Logger, data_name: str, num_route_class: int, embedding_size=128, hidden_size=128,
                 obs_len=20, batch_size=128, dropout=0.2):
        self.net = None
        self.name = data_name
        self.route_class_num = num_route_class
        self.logger = logger
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.obs_len = obs_len
        self.batch_size = batch_size
        self.history = None
        self.dropout = dropout

        self.logger.header('Stage 1: Classification Network on ' + self.name)

    def build_network(self, learning_rate=0.001):
        self.logger.log('Start Build Route Class Classification Net ...')
        left = Sequential()
        left.add(Dense(128, activation='relu', input_shape=(self.obs_len, 2)))

        left.add(
            LSTM(self.hidden_size, input_shape=(self.obs_len, 2), batch_size=self.batch_size, return_sequences=True))
        right = Sequential()
        right.add(Dense(128, activation='relu', input_shape=(self.obs_len, 2)))

        right.add(
            LSTM(self.hidden_size, input_shape=(self.obs_len, 2), batch_size=self.batch_size, return_sequences=True,
                 go_backwards=True))

        self.net = Sequential()
        self.net.add(Merge([left, right], mode='sum'))
        self.net.add(Dropout(self.dropout))
        self.net.add(Conv1D(self.OUT_CHANNEL,
                            self.KERNEL_SIZE,
                            padding='valid',
                            activation='relu',
                            strides=1))
        self.net.add(MaxPooling1D(pool_size=self.POOL_SIZE))
        self.net.add(BatchNormalization())
        self.net.add(Conv1D(self.OUT_CHANNEL,
                            self.KERNEL_SIZE,
                            padding='valid',
                            activation='relu',
                            strides=1))
        self.net.add(MaxPooling1D(pool_size=self.POOL_SIZE))
        self.net.add(Dropout(self.dropout))
        self.net.add(LSTM(self.hidden_size))
        self.net.add(BatchNormalization())
        self.net.add(Dense(self.route_class_num))
        self.net.add(BatchNormalization())
        self.net.add(Activation('softmax'))
        self.net.summary()
        opt = optimizers.RMSprop(lr=learning_rate)
        self.net.compile(optimizer=opt,
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
        self.logger.log('Network Built!')

    def train(self, x, y, epoch=3000):
        self.logger.log('Start Training ...')
        self.logger.field('total training epochs', epoch)

        self.history = self.net.fit([x, x], y,
                                    batch_size=self.batch_size,
                                    epochs=epoch,
                                    validation_split=0,
                                    verbose=0)

    def save(self, save_path, save_name):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_path = save_path + '/' + save_name
        self.net.save(save_path + '.h5')

    def load_trained_model(self, model_path):
        self.net = load_model(model_path)

        self.logger.field('Trained Classification model loaded', model_path)

    def evaluate(self, val_x, val_y, top_n):
        predicted = self.net.predict([val_x, val_x], batch_size=self.batch_size)
        self.logger.field("Evaluate Top ", top_n)
        correct_count = 0
        for i in range(len(val_y)):
            temp_correct = self.evaluate_one(predicted[i], val_y[i], top_n)
            if temp_correct > 0:
                correct_count += 1

        self.logger.field("Classification Net Accuracy", correct_count / len(val_y))

        return correct_count / len(val_y)

    def predict(self, x):
        if len(np.shape(x)) == 2:
            x = np.reshape(x, [-1, self.obs_len, 2])

        if len(np.shape(x)) == 3:
            predicted = self.net.predict([x, x], batch_size=self.batch_size)
            return predicted
        else:
            self.logger.field('Wrong Input Data Shape', np.shape(x))
            self.logger.error('Wrong Input Data Shape for predicting')
            return None

    @staticmethod
    def evaluate_one(predicted: np.array, label: np.array, top_n):
        predicted_index = heapq.nlargest(top_n, range(len(predicted)), predicted.take)
        gt_index = heapq.nlargest(1, range(len(label)), label.take)

        correct_count = 0
        for p in predicted_index:
            if p in gt_index:
                correct_count += 1

        return correct_count

