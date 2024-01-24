# modified 32 to make it weighted knn
# untested
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)


def lp_norm(x, p):
    if p >= 1:
        return np.sum(np.abs(x) ** p) ** (1 / p)


def euclidean_distance(x, y):
    return lp_norm(x - y, 2)


class KNNClassifier:
    def __init__(self, k, weighting_strategy=None) -> None:
        self.k = k
        self.weighting_strategy = weighting_strategy

    def fit(self, xtrain, ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain

    def predict(self, xtest):
        output = [self._predict(x) for x in xtest]
        return np.array(output)

    def get_valid_indices(self, test):
        distances = [euclidean_distance(x, test) for x in self.xtrain]
        distances_indx = np.argsort(distances)[: self.k]
        return distances_indx, np.array(distances)[distances_indx]

    def _predict(self, test):
        distances_indx, distances = self.get_valid_indices(test)
        if self.weighting_strategy is None:
            counter = Counter([self.ytrain[ix] for ix in distances_indx])
        elif self.weighting_strategy == "reciprocal":
            counter = Counter()
            for label, weight in zip(self.ytrain[distances_indx], (1 / distances)):
                counter[label] += weight
        elif self.weighting_strategy == "reciprocal_squared":
            counter = Counter()
            for label, weight in zip(
                self.ytrain[distances_indx], (1 / np.square(distances))
            ):
                counter[label] += weight
        else:
            raise ValueError()
        return counter.most_common(1)[0][0]


class KNNRegression(KNNClassifier):
    def _predict(self, test):
        distances_indx, distances = self.get_valid_indices(test)
        if self.weighting_strategy is None:
            avg = sum([self.ytrain[ix] for ix in distances_indx]) / len(distances_indx)
        elif self.weighting_strategy == "reciprocal":
            weights = 1 / distances
            avg = np.dot(weights, self.ytrain[distances_indx]) / np.sum(weights)
        elif self.weighting_strategy == "reciprocal_squared":
            weights = 1 / np.square(distances)
            avg = np.dot(weights, self.ytrain[distances_indx]) / np.sum(weights)
        else:
            raise ValueError()
        return avg


if __name__ == "__main__":
    knn = KNNClassifier(3, weighting_strategy="reciprocal")
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    acc = np.mean(preds == y_test)
    print("Accuracy :", acc)
