import numpy as np
from entropy_IG import Information_gain

class node:
    def __init__(self,feature=None, threshold=None,left=None,right=None,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

#Function for Partition
def partition(X,y,feature_index,threshold):
    left_mask = X[:,feature_index] <= threshold #left mask and right mask will contain boolean expressions for the row "feature_index" for the given condition
    right_mask = X[:,feature_index] > threshold
    X_left = X[left_mask] #All the rows in X which has "True" value in "feature_index" column for the left mask will be selected
    X_right = X[right_mask] #All the rows in X which has "True" value in "feature_index" column for the right mask will be selected
    y_left = y[left_mask] #All the rows in y for which there was corresponding true values in x will be selected
    y_right = y[right_mask]
    return X_left , X_right , y_left , y_right

#Function to decide best split
def best_split(X,y):
    best_gain = 0
    best_feature = None
    best_threshold = None
    rows , columns = X.shape

    for feature_index in range (columns): #Looping through each feature to find the best feature
        sorted_indices = np.argsort(X[:,feature_index]) #Sorting the indices of the feature values in ascending order
        for j in range (rows - 1):
            threshold = (X[sorted_indices[j], feature_index] + X[sorted_indices[j + 1], feature_index]) / 2
            X_left,X_right,y_left,y_right = partition(X,y,feature_index,threshold)
            if len(y_left) == 0 or len(y_right) == 0: #If the threshold yields only one-sided tree, then we skip the threshold
                continue
            else:
                gain = Information_gain(y,y_left,y_right) #Calculation of information gain
            if gain > best_gain:
                best_gain , best_feature , best_threshold = gain , feature_index , threshold
    return best_feature , best_threshold


#Function for building a decision tree
def build_tree(X,y,depth=0,max_depth=5):
    # 1.To check if the stopping conditions are met
    unique_classes = np.unique(y) # Checking if all the points in the leaf/node belong to the same class
    if len(unique_classes) == 1 or depth >= max_depth: #If all the points in the node/leaf belong to the same class we stop building a tree for that node
        return node(value = np.bincount(y).argmax())

    # 2. Find the best feature and threshold
    feature , threshold = best_split(X, y)

    # 3. If no valid split is found, return the node value
    if feature is None: # If no feature is found good, there will be only 1 leaf
        return node(value = np.bincount(y).argmax())  # and the leaf will contain

    # 4. Partition data into left and right split
    X_left , X_right , y_left , y_right = partition(X , y , feature , threshold)

    # Step 5: Recursively build left and right subtrees
    left_subtree = build_tree(X_left, y_left, depth + 1, max_depth)
    right_subtree = build_tree(X_right, y_right, depth + 1, max_depth)

    # Step 6: Return a node with the selected feature and threshold
    return node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)
