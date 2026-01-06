import numpy as np


def mse(y, y_hat):
    mse = 1/(2 * y.size) * np.sum((y_hat - y) ** 2)
    return mse


class LinearRegression:
    def __init__(self):
        # initialize w and b with random values
        self.w = np.random.random(size=1)
        self.b = np.random.random(size=1)

    def fit(self, X, y, alpha=0.01, epochs=50):
        # apply gradient descent as an optimizer to train the model
        for i in range(epochs):
            # simultanous update of the parameters w and b
            m = len(X)
            tmp_w = self.w - alpha * \
                (1/m) * np.sum(((self.w * X + self.b) - y) * X)
            tmp_b = self.b - alpha * 1/m * np.sum((self.w * X + self.b) - y)
            self.w = tmp_w
            self.b = tmp_b
        return None

    def predict(self, x):
        y_hat = self.w * x + self.b
        return y_hat
