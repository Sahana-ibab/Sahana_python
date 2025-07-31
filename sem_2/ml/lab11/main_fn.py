# Implement decision tree classifier without using scikit-learn using
# the iris dataset. Fetch the iris dataset from scikit-learn library.

import numpy as np
from sklearn import datasets
import pandas as pd

from build_tree_algo import build_tree
from prediction_algo import predict_batch
from print_tree_algo import print_tree

#  Function to Load dataset:
def load_dataset():
    df = datasets.load_iris()
    # X = pd.DataFrame(df.data, columns=df.feature_names)  # Convert X to DataFrame
    # y = pd.Series(df.target, name="target")  # Convert y to Series
    X = df.data
    y = df.target
    return X, y


def main():
    X, y = load_dataset()

    # Train the decision tree
    tree = build_tree(X, y, max_depth=5)

    # Test on training data
    predictions = predict_batch(tree, X)
    accuracy = np.mean(predictions == y) * 100
    print("Decision Tree Accuracy:", accuracy, "%")

    # Print the tree structure
    print_tree(tree)

if __name__ == '__main__':
    main()




