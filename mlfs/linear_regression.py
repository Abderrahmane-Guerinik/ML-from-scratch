import numpy as np


def mse(y, y_hat):
    mse = 1/(2 * y.size) * np.sum((y_hat - y) ** 2)
    return mse


class LinearRegression:
    def __init__(self):
        self.w = 0
        self.b = 0

    def fit(X, y):
        return None

    def predict(self, x):
        y_hat = self.w * x + self.b
        return y_hat
