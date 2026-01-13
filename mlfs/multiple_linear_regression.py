import numpy as np
import pandas as pd


""" def mse(y, y_hat):
    mse = 1/(2 * y.size) * np.sum((y_hat - y) ** 2)
    return mse """


class MultipleLinearRegression:
    def __init__(self):
        # initialize w and b with random values
        self.w = 0
        self.b = 0

    def fit(self, X, y, alpha=0.01, epochs=50, threshold=1e-6, patience=10):
        # inititalisation of w
        self.w = np.zeros(shape=(X.shape[1],))
        # apply gradient descent as an optimizer to train the model
        loss = float("inf")
        count = 0
        m = len(X)
        for epoch in range(epochs):
            # simultanous update of the parameters w and b 
            tmp_w = self.w - alpha * 1/m * X.T @ (X @ self.w + self.b - y)
            tmp_b = self.b - alpha * 1/m * np.sum((X @ self.w + self.b - y))
            self.w = tmp_w
            self.b = tmp_b
            # track the loss function
            prev_loss = loss
            loss = 1/(2*m) * np.sum((X @ self.w + self.b - y) ** 2)
            if epoch % 10 == 0:
                print(f"Epoch {epoch} loss: {loss} w: {self.w} b: {self.b}")
            if abs(loss - prev_loss) <= threshold:
                count += 1
                if count == patience:
                    print(
                        f"Early stopping at epoch: {epoch} final w: {self.w} final b: {self.b}.")
                    break
            else:
                count = 0
        return None

    def predict(self, x):
        y_hat = np.dot(self.w, x) + self.b
        return y_hat
