import numpy as np

def entropy_cal(y):
    entropy = 0
    unique_classes, counts = np.unique(y, return_counts=True)
    probabilities = counts/len(y)
    for p in probabilities:
        if p > 0:
            entropy+=-p * (np.log2(p))
    return entropy

def information_gain(y, y_left, y_right):
    y_entropy = entropy_cal(y)
    y_right_entropy = entropy_cal(y_right)
    y_left_entropy = entropy_cal(y_left)

    weight_y_right = len(y_right)/len(y)
    weight_y_left = len(y_left)/len(y)

    IG =y_entropy-(( weight_y_right * y_right_entropy ) + ( weight_y_left * y_left_entropy))
    return IG

