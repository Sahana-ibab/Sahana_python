import pandas as pd
import numpy as np
import math

# Function to compute the entropy
def entropy(y):
    entropy_value = 0
    unique_classes , counts = np.unique( y, return_counts = True ) # First we find the unique values and the counts corresponding to them
    probabilities = counts / len(y) # probabilities is a list which contains the counts
    for p in probabilities: # Calculating the entropy
        if p > 0:
            entropy_value += -p * math.log2(p)
    return entropy_value

# Computing the information gain
def Information_gain( y, y_left, y_right ):
    H_parent = entropy(y)
    H_left = entropy(y_left)
    H_right = entropy(y_right)
    weight_left = len(y_left) / len(y)    # weight --> (n+p/total)
    weight_right = len(y_right) / len(y)
    H_children = ( weight_left * H_left ) + ( weight_right * H_right )
    IG = H_parent - H_children

    return IG




























# def cal_entropy(labels):
#     total_count = len(labels)
#     label_counts = Counter(labels)
#
#     entropy = 0.0
#     for count in label_counts.values():
#         probability = count / total_count
#         entropy -= probability * math.log2(probability)
#     return entropy
#
#
# data =  pd.read_csv('/home/ibab/breast_cancer_row.csv')
#
# # X = data.iloc[:, :-1]
# # y =  data.iloc[:,-1]
# X = data.iloc[:, :-1].astype(str)
# y = data.iloc[:, -1].astype(str)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=20)
# # print(X_train.shape)
# le = LabelEncoder()
# le.fit(y_train)
# y_train = le.transform(y_train)
# y_test = le.transform(y_test)
#
# print("Entropy (Train Set):", cal_entropy(y_train))
# print("Entropy (Test Set):", cal_entropy(y_test))
