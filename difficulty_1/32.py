from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)


def euclidean_distance(x, y):
    return np.sqrt(np.sum(np.abs(x - y) ** 2))


class KNN:
    def __init__(self, k) -> None:
        self.k = k

    def fit(self, xtrain, ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain

    def predict(self, xtest):
        output = [self._predict(x) for x in xtest]
        return np.array(output)

    def _predict(self, test):
        distances = [euclidean_distance(x, test) for x in self.xtrain]
        distances_indx = np.argsort(distances)[: self.k]
        most_common = Counter([self.ytrain[ix] for ix in distances_indx]).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    knn = KNN(3)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    acc = np.mean(preds == y_test)
    print("Accuracy :", acc)
