import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(x))

class LogicticRegression:
    def __init__(self, threshold=0.5):
        # initialise the threshold & bias
        self.threshold = threshold
        self.b = 0
        return None 
    
    def fit(self, X, y, epochs=100, alpha=0.001, threshold=1e-6, patience=10):
        # set the weights to zero
        self.w = np.zeros(X.shape[1])
        m = len(X) # number of instances
        cost = float("inf")
        count = 0
        for epoch in range(epochs):
            # simultanous update of w & b
            tmp_w = self.w - alpha * 1/m * X.T @ (X @ self.w + self.b - y)
            tmp_b = self.b - alpha * 1/m * np.sum((X @ self.w + self.b) - y)
            self.w = tmp_w 
            self.b = tmp_b
            
            prev_cost = cost
            cost = -1/m * (np.dot(y.T, np.log(y_hat)) + np.dot((1-y).T, np.log(1 - y_hat)))
            # display the cost & params each 10 epochs
            if epoch % 10 == 0:
                y_hat = X @ self.w + self.b
                y_hat = 0 if y_hat < self.threshold else 1
                print(f"Epoch: {epoch} Cost: {cost} w: {self.w} b: {self.b}")
            
            if abs(cost - prev_cost) < threshold:
                count_patience+=1
                if count == patience:
                    print(f"Training stoped at epoch {epoch} cost:{cost} final w: {self.w} final b: {self.b}")
                    break
            else:
                count = 0
        return None 
    
    def predict(self, x, y):
        # z = w.x + b
        # y_hat = sigmoid(z)
        z = np.dot(self.w, x) + self.b
        if sigmoid(z) >= self.threshold:
            return 1
        else:
            return 0