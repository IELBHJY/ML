from sklearn.datasets import load_boston
import numpy as np


class MyLinearRegression:
    def __init__(self):
        self.data = load_boston()
        self.X = self.data.data
        self.y = self.data.target
        self.w = np.zeros([self.X.shape[1]]) + 0.01
        self.b = 0.01

    def fit(self, x=None, y=None):
        if x is not None and y is not None:
            self.X = x
            self.y = y
        ones = np.ones([self.X.shape[0]])
        X = np.column_stack((self.X, ones))
        res = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), self.y.T)
        self.w = res[0:-1]
        self.b = res[-1]

    def predict(self, X):
        y_pre = np.matmul(self.w, X) + self.b
        return y_pre


if __name__ == '__main__':
    lr = MyLinearRegression()
    lr.fit()

