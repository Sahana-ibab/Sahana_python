# Implement a decision regression tree algorithm without using
# scikit-learn using the diabetes dataset. Fetch the dataset from
# scikit-learn library.

#  only changes in DTR compared to DTC:
#   MSE is cal, instead of Ig
#   Best_split fn
#   Stop splitting when variance is low
#   store mean target value at leaf nodes

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Node class for the decision tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


# Load diabetes dataset
def load_dataset():
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    return X, y

# Function to partition data based on threshold
def partition(X, y, feature, threshold):
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

# Function to find the best split based on Mean squared error (MSE)
def best_split(X, y):
    best_mse = float('inf')    # because it should minimize
    best_feature = None
    best_threshold = None
    rows, columns = X.shape

    for feature_index in range(columns):
        sorted_indices = np.argsort(X[:, feature_index])  # Sort feature values
        for j in range(rows - 1):
            threshold = (X[sorted_indices[j], feature_index] + X[sorted_indices[j + 1], feature_index]) / 2
            X_left, X_right, y_left, y_right = partition(X, y, feature_index, threshold)

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            # Compute weighted variance (MSE)
            mse = (np.var(y_left) * len(y_left) + np.var(y_right) * len(y_right)) / len(y)

            if mse < best_mse:  # Minimize MSE
                best_mse, best_feature, best_threshold = mse, feature_index, threshold

    return best_feature, best_threshold

# Function to build the decision tree
def build_tree(X, y, depth=0, max_depth=5):
    if len(y) == 0:
        return None

    # Stop if max depth is reached or variance is too small
    if depth >= max_depth or np.var(y) < 1e-5:
        return Node(value=np.mean(y))  # Leaf node stores mean target value

    feature, threshold = best_split(X, y)

    if feature is None:
        return Node(value=np.mean(y))

    X_left, X_right, y_left, y_right = partition(X, y, feature, threshold)

    left_subtree = build_tree(X_left, y_left, depth + 1, max_depth)
    right_subtree = build_tree(X_right, y_right, depth + 1, max_depth)

    return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)

# Function to predict a single data point
def predict(node, X):
    if node.value is not None:  # Leaf node stores mean value
        return node.value
    if X[node.feature] <= node.threshold:
        return predict(node.left, X)
    else:
        return predict(node.right, X)

# Function to predict multiple samples
def predict_batch(tree, X):
    return np.array([predict(tree, sample) for sample in X])


def main():
    X, y = load_dataset()

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the decision tree regressor
    tree = build_tree(X_train, y_train, max_depth=5)

    # Predict on test set
    y_pred = predict_batch(tree, X_test)

    # Compute Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print("Decision Tree Regression MSE:", mse)


if __name__ == '__main__':
    main()
