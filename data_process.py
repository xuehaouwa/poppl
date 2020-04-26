from copy import deepcopy
import numpy as np
import os
from sklearn.cluster import KMeans
from gv_tools.util.logger import Logger


class DataProcessor:
    def __init__(self, data_path, logger: Logger, obs_len=20, pred_len=20):

        self.source_points = None
        self.end_points = None
        self.source_point_class_label = None
        self.end_point_class_label = None
        self.obs_len = obs_len
        self.pred_len = pred_len

        self.logger = logger
        # self.classification_train_data = self.data[0: self.training_num, :, :]
        self.classification_train_data = np.concatenate((np.load(os.path.join(data_path, "orig_train_obs.npy")),
                                                         np.load(os.path.join(data_path, "orig_train_pred.npy"))), axis=1)
        self.training_num = len(self.classification_train_data)
        self.logger.field("Data Used for Classification Net Training Shape", np.shape(self.classification_train_data))
        self.preprocess()
        self._num_route_class = None
        self.total_sub_x = None
        self.final_test_x = np.load(os.path.join(data_path,  "orig_test_obs.npy"))
        self.final_test_y = np.load(os.path.join(data_path,  "orig_test_pred.npy"))
        self.total_sub_y = None
        self.total_classification_y = None
        self.one_hot_label_dic = None
        self.rc_index_dic: dict = {}
        self.rc_count=[]

    def preprocess(self):
        self.source_points = self.classification_train_data[:, 0, :]
        self.end_points = self.classification_train_data[:, -1, :]
        self.logger.field("Source Points Data Shape", np.shape(self.source_points))
        self.logger.field("Ending Points Data Shape", np.shape(self.end_points))

    def process_for_classification(self, num_pc, val_split=0.2, min_num_in_rc_percent=0.05, min_num_in_rc=100):

        # points clustering
        self.class_points(num_pc=num_pc)
        # min_num_in_rc = int(self.training_num * min_num_in_rc_percent)
        min_num_in_rc = min_num_in_rc
        rc_label_list, max_rc_number_index = self.get_rc_list(num_pc=num_pc, min_num_in_rc=min_num_in_rc)
        rc_one_hot_label = self.generate_one_hot_label(rc_label_list)
        self.one_hot_label_dic = dict((repr(v),k) for k,v in rc_one_hot_label.items())
        val_num = int(val_split * self.training_num)
        x, y = self.generate_classification_data(rc_one_hot_label=rc_one_hot_label, rc_label_list=rc_label_list)
        self.total_classification_y = y
        train_x = x[val_num:]
        train_y = y[val_num:]
        val_x = x[0: val_num]
        val_y = y[0: val_num]
        aug_train_x = self.reverse(self.total_sub_y[val_num:], return_concated=True)
        aug_train_y = np.concatenate((train_y, train_y), axis=0)
        self.logger.field("Train Data X for Classification Net Shape", np.shape(train_x))
        self.logger.field("Train Data Y for Classification Net Shape", np.shape(train_y))
        self.logger.field("VAl Data X for Classification Net Shape", np.shape(val_x))
        self.logger.field("VAL Data Y for Classification Net Shape", np.shape(val_y))
        return train_x, train_y, val_x, val_y, max_rc_number_index, aug_train_x, aug_train_y

    def generate_classification_data(self, rc_one_hot_label, rc_label_list):
        x = []
        y = []
        self.total_sub_x = []
        self.total_sub_y = []

        for i in range(self.training_num):
            if [self.source_point_class_label[i], self.end_point_class_label[i]] in rc_label_list:
                x.append(self.classification_train_data[i][0: self.obs_len])
                self.total_sub_x.append(self.classification_train_data[i][0: self.obs_len])
                self.total_sub_y.append(self.classification_train_data[i][self.obs_len: ])
                y.append(rc_one_hot_label.get(str([self.source_point_class_label[i], self.end_point_class_label[i]])))
            elif [self.end_point_class_label[i], self.source_point_class_label[i]] in rc_label_list:
                x.append(self.classification_train_data[i][0: self.obs_len])
                self.total_sub_x.append(self.classification_train_data[i][0: self.obs_len])
                self.total_sub_y.append(self.classification_train_data[i][self.obs_len:])
                y.append(rc_one_hot_label.get(str([self.end_point_class_label[i], self.source_point_class_label[i]])))

        x = np.reshape(x, [-1, self.obs_len, 2])
        y = np.reshape(y, [-1, len(rc_label_list)])
        self.total_sub_x = np.reshape(self.total_sub_x, [-1, self.obs_len, 2])
        self.total_sub_y = np.reshape(self.total_sub_y, [-1, self.pred_len, 2])
        self.logger.field("Total Classification X for Classification Net Shape", np.shape(x))
        self.logger.field("Total Classification Y for Classification Net Shape", np.shape(y))
        self.logger.field("Total Sub Net Training X Shape", np.shape(self.total_sub_x))
        self.logger.field("Total Sub Net Training Y Shape", np.shape(self.total_sub_y))
        return x, y

    def count_route_class(self, source_cluster, destination_cluster):
        count = 0
        for i in range(len(self.source_point_class_label)):
            if (self.source_point_class_label[i] == source_cluster) \
                    and (self.end_point_class_label[i] == destination_cluster):
                count += 1

        return count

    def class_points(self, num_pc):
        self.logger.log("Start Points Clustering ...")
        self.logger.field("Number of points cluster", num_pc)
        all_points = np.concatenate((self.source_points, self.end_points), axis=0)
        km = KMeans(n_clusters=num_pc, random_state=0).fit(all_points)
        labels = km.labels_
        self.source_point_class_label = labels[0: self.training_num]
        self.end_point_class_label = labels[self.training_num:]
        self.logger.field("Source Points Class Label Shape", np.shape(self.source_point_class_label))
        self.logger.field("Ending Points Class Label Shape", np.shape(self.end_point_class_label))

    def get_rc_list(self, num_pc, min_num_in_rc):
        rc_label_list = []
        num_rc = 0
        total_number_with_rc = 0

        rc_label_num_list = []

        for source_label in range(num_pc):
            for end_label in range(num_pc):
                if source_label < end_label:
                    count = self.count_route_class(source_label, end_label) + \
                            self.count_route_class(end_label, source_label)
                    self.rc_count.append(count)
                    if count >= min_num_in_rc:
                        total_number_with_rc += count
                        num_rc += 1
                        rc_label_list.append([source_label, end_label])
                        rc_label_num_list.append(count)
                        self.logger.field("Route Class", [source_label, end_label])
                elif source_label == end_label:
                    count = self.count_route_class(source_label, end_label)
                    self.rc_count.append(count)
                    if count >= min_num_in_rc:
                        total_number_with_rc += count
                        num_rc += 1
                        rc_label_list.append([source_label, end_label])
                        rc_label_num_list.append(count)
                        self.logger.field("Route Class", [source_label, end_label])



        self.logger.field("Total Number of Route Classes", len(rc_label_list))
        self._num_route_class = num_rc
        max_rc_number_index = rc_label_num_list.index(max(self.rc_count))
        print(rc_label_num_list)
        self.logger.field("Total Number of Trajectories in Route Classes", total_number_with_rc)
        self.logger.field("Sorted RC numbers", sorted(self.rc_count))
        return rc_label_list, max_rc_number_index

    def generate_sub_net_training(self, route_class_index: int):

        sub_x = []
        sub_y = []
        rc_one_hot = self.rc_index_dic[route_class_index]
        rc_name = self.one_hot_label_dic[repr(rc_one_hot)]
        route_class_name = 'rc_' + list(rc_name)[1] + '_' + list(rc_name)[-2]
        self.logger.field("Generating Training Data for Sub Net", route_class_name)

        for i in range(len(self.total_classification_y)):
            if np.argmax(self.total_classification_y[i]) == route_class_index:
                sub_x.append(self.total_sub_x[i])
                sub_y.append(self.total_sub_y[i])

        sub_x = np.reshape(sub_x, [-1, self.obs_len, 2])
        sub_y = np.reshape(sub_y, [-1, self.pred_len, 2])
        self.logger.field("Sub Net X Shape", np.shape(sub_x))
        self.logger.field("Sub Net Y Shape", np.shape(sub_y))

        return sub_x, sub_y, route_class_name

    def generate_one_hot_label(self, rc_label_list):
        rc_one_hot_label = {}
        for i in range(len(rc_label_list)):
            on_hot = [0] * len(rc_label_list)
            on_hot[i] = 1
            rc_one_hot_label[repr(rc_label_list[i])] = on_hot
            self.rc_index_dic[i] = on_hot

        return rc_one_hot_label

    def get_final_testing_data(self):
        self.logger.field('Final Test X Shape', np.shape(self.final_test_x))
        self.logger.field('Final Test Y Shape', np.shape(self.final_test_y))
        return self.final_test_x, self.final_test_y

    @property
    def rc_number(self):
        return self._num_route_class

    @staticmethod
    def reverse(trajs, return_concated=False):
        """
        reverse trajectories
        :param trajs:
        :return:
        """
        reversed_trajs = deepcopy(trajs)
        reversed_trajs = np.flip(reversed_trajs, 1)

        if return_concated:
            reversed_trajs = np.concatenate((reversed_trajs, trajs), axis=0)

        return reversed_trajs


