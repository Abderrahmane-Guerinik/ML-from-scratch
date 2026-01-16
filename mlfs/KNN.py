import numpy as np
from collections import Counter

class KNN():
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X = X
        self.y = y 
    
    def predict(self, x):
        distances = np.linalg.norm(self.X - x, axis=1)
        k_nearest_neighbors_indices = np.argsort(distances)[:self.k]
        k_nearest_neighbors_y = self.y[k_nearest_neighbors_indices]
        counts = Counter(k_nearest_neighbors_y)
        return counts.most_common(1)[0][0]