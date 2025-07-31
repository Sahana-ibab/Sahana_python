import numpy as np

def predict(node, X):
    if node.value is not None: # Only leaf node will have values
        return node.value
    if X[node.feature] <= node.threshold:
        return predict(node.left, X)
    else:
        return predict(node.right, X)

# Function to classify multiple samples
def predict_batch(tree, X):
    return np.array([predict(tree, sample) for sample in X]) #Loops through each row in the sample and carries out predict function for each sample
