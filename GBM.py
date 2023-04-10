import numpy as np

class GBM:
    def __init__(self, n_trees, max_depth, learning_rate):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees = []
        self.predictions = None

    def fit(self, X, y):
        self.predictions = np.full(X.shape[0], np.mean(y))
        for i in range(self.n_trees):
            tree = self.build_tree(X, y - self.predictions, depth=0)
            self.trees.append(tree)
            self.predictions += self.learning_rate * self.predict(tree, X)

    def build_tree(self, X, y, depth):
        if depth == self.max_depth or len(y) == 0:
            return np.mean(y)
        else:
            split_feature, split_value = self.choose_split(X, y)
            left_idx = X[:, split_feature] < split_value
            right_idx = X[:, split_feature] >= split_value
            left_tree = self.build_tree(X[left_idx], y[left_idx], depth+1)
            right_tree = self.build_tree(X[right_idx], y[right_idx], depth+1)
            return (split_feature, split_value, left_tree, right_tree)

    def choose_split(self, X, y):
        min_error = float('inf')
        best_feature = None
        best_value = None
        for feature in range(X.shape[1]):
            for value in np.unique(X[:, feature]):
                left_idx = X[:, feature] < value
                right_idx = X[:, feature] >= value
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                left_y = y[left_idx]
                right_y = y[right_idx]
                error = np.sum((left_y - np.mean(left_y))**2) + np.sum((right_y - np.mean(right_y))**2)
                if error < min_error:
                    min_error = error
                    best_feature = feature
                    best_value = value
        return best_feature, best_value

    def predict(self, tree, X):
        if type(tree) == float:
            return np.full(X.shape[0], tree)
        else:
            split_feature, split_value, left_tree, right_tree = tree
            left_idx = X[:, split_feature] < split_value
            right_idx = X[:, split_feature] >= split_value
            pred = np.zeros(X.shape[0])
            pred[left_idx] = self.predict(left_tree, X[left_idx])
            pred[right_idx] = self.predict(right_tree, X[right_idx])
            return pred

