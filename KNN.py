import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        
        # Find the indices of the k nearest neighbors using a manual approach
        k_indices = self._get_k_indices(distances)
        
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
    
    def _get_k_indices(self, distances):
        k_indices = []
        for _ in range(self.k):
            min_index = 0
            min_distance = distances[0]
            for i, distance in enumerate(distances):
                if distance < min_distance:
                    min_distance = distance
                    min_index = i
            k_indices.append(min_index)
            distances[min_index] = np.inf
        return k_indices

# Example usage
if __name__ == "__main__":
    X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[2, 2], [4, 5]])

    model = KNN(k=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Test data:", X_test)
    print("Predicted labels:", y_pred)
