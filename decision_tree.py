
import numpy as np
import pandas as pd
import sklearn
import copy

MAX_DEPTH = 5
MIN_DEPTH = 3


class decision_tree:
    def __initialize_assigned_classes(self, i):
        self.tree[2 * i + 1] = str(np.random.choice(['1', '0'], p=[0.6, 0.4]))
        self.tree[2 * i + 2] = str(np.random.choice(['1', '0'], p=[0.4, 0.6]))

    def _initialize_features(self, i, depth):
        self.tree[2 * i + 1] = np.random.randint(0, self.number_of_features - depth)
        self.tree[2 * i + 2] = self.tree[2 * i + 1]

        while self.tree[2 * i + 2] == self.tree[2 * i + 1]:
            self.tree[2 * i + 2] = np.random.randint(0, self.number_of_features - depth)

        if depth < MIN_DEPTH - 1:
            self._initialize_features(2 * i + 1, depth + 1)
            self._initialize_features(2 * i + 2, depth + 1)

        elif depth >= MAX_DEPTH - 1:
            self.__initialize_assigned_classes(2 * i + 1)
            self.__initialize_assigned_classes(2 * i + 2)

        else:
            classes_or_features = bool(np.random.choice([True, False]))

            if classes_or_features:
                self.__initialize_assigned_classes(2 * i + 1)
                self.__initialize_assigned_classes(2 * i + 2)
            else:
                self._initialize_features(2 * i + 1, depth + 1)
                self._initialize_features(2 * i + 2, depth + 1)

    def _start_initialization_of_tree(self):
        self.tree[0] = np.random.randint(0, self.number_of_features)

        self._initialize_features(i=0, depth=1)

    def __init__(self, data, features):
        self.tree = [None] * (2 ** (MAX_DEPTH + 1) - 1)

        self.means = data[data['class'] == 1][features].mean()
        self.features = features
        self.number_of_features = len(features)

        self._start_initialization_of_tree()

    def __str__(self):
        s = "tree: [ "
        for element in self.tree:
            s += str(element) + " "

        s += "]"

        return s

    def classify_mail(self, instance):
        features_at_depth = copy.deepcopy(self.features)
        position = 0

        while True:
            feature_index = self.tree[position]
            get_feature = features_at_depth[feature_index]
            features_at_depth.remove(get_feature)

            if instance[get_feature] > self.means[get_feature]:
                position = 2 * position + 1

            else:
                position = 2 * position + 2

            if type(self.tree[position]) is str:
                break

        return int(self.tree[position])