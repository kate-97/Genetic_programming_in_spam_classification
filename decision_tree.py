
import numpy as np
import pandas as pd
import sklearn
import copy


class decision_tree:
    def _initialize_features(self, i, depth):
        self.tree[2 * i + 1] = np.random.randint(0, self.number_of_features - depth)
        self.tree[2 * i + 2] = self.tree[2 * i + 1]

        while self.tree[2 * i + 2] == self.tree[2 * i + 1]:
            self.tree[2 * i + 2] = np.random.randint(0, self.number_of_features - depth)

        if depth >= 3:
            return

        else:
            self._initialize_features(2 * i + 1, depth + 1)
            self._initialize_features(2 * i + 2, depth + 1)

    def _initialize_assigned_classes(self):
        self.tree[15] = np.random.choice([1, 0], p=[0.6, 0.4])
        # self.tree[15] = np.random.choice([1, 0], p=[0.8, 0.2])
        # self.tree[30] = np.random.choice([1, 0], p=[0.2, 0.8])

        for i in range(16, 30):
            self.tree[i] = np.random.choice([1, 0])

        self.tree[30] = np.random.choice([1, 0], p=[0.4, 0.6])

    def _start_initialization_of_tree(self):
        self.tree[0] = np.random.randint(0, self.number_of_features)

        self._initialize_features(0, 1)

        self._initialize_assigned_classes()

    def __init__(self, features):
        self.tree = [None] * (2 ** 5 - 1)
        self.features = features
        self.number_of_features = len(features)

        self._start_initialization_of_tree()

    def __str__(self):
        s = "tree: [ "
        for element in self.tree:
            s += str(element) + " "

        s += "]"

        return s

    def classify_mail(self, instance, means):
        features_at_depth = copy.deepcopy(self.features)
        position = 0

        while position < 15:
            feature_index = self.tree[position]
            get_feature = features_at_depth[feature_index]
            features_at_depth.remove(get_feature)

            if instance[get_feature] > means[get_feature]:
                position = 2 * position + 1

            else:
                position = 2 * position + 2

        return self.tree[position]