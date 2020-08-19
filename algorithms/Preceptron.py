from sklearn.datasets import load_breast_cancer
import time
import numpy as np
import random

data = load_breast_cancer()


class Timer:
    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self):
        self.end_time = time.time()
        print("This model running time:", self.end_time - self.start_time)


class MyPerception:
    def __init__(self, step, max_iteration_num):
        self.lr = step
        self.max_iteration_num = max_iteration_num
        self.y_pre =None

    def train(self, X, y):
        if len(X) == 0:
            print("input data shape is zero!")
            return
        feature_len = len(X[0])
        data_len = len(X)
        fun1 = np.frompyfunc(lambda x: -1 if x == 0 else 1, 1, 1)
        y = fun1(y)
        self.b_ = 0
        self.w_ = np.zeros(feature_len)
        self.w_ = self.w_.reshape(1, len(self.w_))
        loop_num = 0
        while loop_num < self.max_iteration_num:
            select_index = random.randint(0, len(X) - 1)
            y_predict = np.dot(self.w_, X[select_index].T) + self.b_
            if y_predict * y[select_index] <= 0:
                self.w_ = self.w_ + self.lr * y[select_index] * X[select_index]
                self.b_ = self.b_ + self.lr * y[select_index]
            loop_num += 1

        y_predict = np.dot(X, self.w_.T) + self.b_
        y_res = np.array([y_predict[index][0] * y[index] for index in range(data_len)])
        train_indexs = np.where(y_res < 0)
        if len(train_indexs[0]) == 0:
            print("All true!!!")
        else:
            print("wrong data len = ", len(train_indexs[0]))

    def predict(self, X):
        self.y_pre = np.dot(X, self.w_.T) + self.b_
        return [1 if value > 0 else 0 for value in self.y_pre]

    def score(self, X, y):
        self.y_pre = np.dot(X, self.w_.T) + self.b_
        label = [1 if value > 0 else 0 for value in self.y_pre]
        if len(label) != len(y):
            raise print("input data shape of X != shape of y")
        return sum(label == y) / len(y)


perception = MyPerception(0.1, 10000)
perception.train(data.data, data.target)

print("************MyPerception*********************")
print(perception.w_)
print(perception.b_)
print(perception.score(data.data, data.target))
print("************MyPerception*********************")

print("************sklearnPerception*********************")
from sklearn.linear_model import Perceptron

model = Perceptron()
model.fit(data.data, data.target)

print(model.coef_)
print(model.intercept_)
print(model.score(data.data, data.target))
print("************sklearnPerception*********************")
