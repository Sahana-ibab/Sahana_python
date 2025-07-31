import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_iris


# Entropy function (same as before)
def entropy(data):
    counts = Counter(data)
    n = sum(counts.values())
    probabilities = [count / n for count in counts.values()]
    entropy = -np.sum([p * np.log2(p) for p in probabilities])
    return entropy


# Weighted entropy function
def weighted_entropy(left_child, right_child):
    n_total = len(left_child) + len(right_child)
    left_total = len(left_child)
    right_total = len(right_child)

    H_left = entropy(left_child)
    H_right = entropy(right_child)

    return (H_left * (left_total / n_total)) + (H_right * (right_total / n_total))


# Information Gain function
def ig(parent, left_child, right_child):
    H_parent = entropy(parent)
    H_weighted = weighted_entropy(left_child, right_child)
    IG = H_parent - H_weighted
    return IG


# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target  # Target labels (0, 1, 2)

# Simulate a split (for example, based on petal length)
split_feature = 'petal length (cm)'
split_value = 2.5  # A chosen threshold

# Create the parent, left_child, and right_child
parent = data['target'].tolist()
left_child = data[data[split_feature] <= split_value]['target'].tolist()
right_child = data[data[split_feature] > split_value]['target'].tolist()

# Compute information gain
information_gain = ig(parent, left_child, right_child)
print(f"INFORMATION GAIN: {information_gain:.4f}")
