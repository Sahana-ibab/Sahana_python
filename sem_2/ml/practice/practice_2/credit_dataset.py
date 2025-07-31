import pandas as pd
import numpy as np
from collections import Counter

def entropy(data):
    counts = Counter(data)
    n = sum(counts.values())
    probabilities = [count / n for count in counts.values()]
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy

def weighted_entropy(left_child, right_child):
    n_total = len(left_child) + len(right_child)
    H_left = entropy(left_child)
    H_right = entropy(right_child)
    return (H_left * len(left_child) / n_total) + (H_right * len(right_child) / n_total)

def information_gain(parent, left_child, right_child):
    H_parent = entropy(parent)
    H_weighted = weighted_entropy(left_child, right_child)
    return H_parent - H_weighted

# Load dataset
df = pd.read_csv("/mnt/data/Credit.csv")

# Selecting a categorical column for entropy calculation (modify as needed)
categorical_column = "Student"  # Example categorical column
parent = df[categorical_column].tolist()
entropy_value = entropy(parent)
print(f"Entropy of {categorical_column}: {entropy_value:.4f}")

# Selecting a numerical feature and split value for information gain calculation
split_feature = "Balance"  # Example feature
split_value = df[split_feature].median()  # Example split at median

# Splitting data
left_child = df[df[split_feature] <= split_value][categorical_column].tolist()
right_child = df[df[split_feature] > split_value][categorical_column].tolist()

# Calculate Information Gain
info_gain = information_gain(parent, left_child, right_child)
print(f"Information Gain for splitting {split_feature} at {split_value}: {info_gain:.4f}")
