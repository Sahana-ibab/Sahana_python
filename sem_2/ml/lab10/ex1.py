import pandas as pd
import numpy as np

# Calculating Entropy:
def entropy(feature):
    # calculating the probabilities ----> shape[0] because 2d
    val = feature.value_counts()/feature.shape[0]

    # calculating the entropy with formula
    entropy = np.sum(-val*np.log2(val+1e-9))
    return entropy

# function to calculate information gain:
def info_gain(feature, leaf):

    # sum is used here as labels are 0s & 1s, because
    # when we add only number of 1s will be accounted...
    sum_a= sum(leaf)
    # print(sum_a)
    # print(leaf.shape[0])
    b = leaf.shape[0] - sum_a   # total - number of 1s
    if sum_a == 0 or b == 0 :  # if either of 2 labels has 0 occurrences
        information_g = 0
    else:
        # != 'O'--> this is to check whether the feature is numerical or not..
        # pandas datatype--> In pandas, the dtype 'O' (object)
        # is typically used for categorical or string data.
        if feature.dtypes != 'O':
            information_g = feature.var() - (sum_a/(sum_a+b)* feature[leaf]).var() - (b/(sum_a+b)*feature[-leaf]).var()

        # for categorical features:
        else:
            information_g = entropy(feature)-sum_a/(sum_a+b)*entropy(feature[-leaf])
    return information_g

def main():
    dt = pd.read_csv("/home/ibab/data/titanic.csv")
    # dt = pd.read_csv("titanic.csv")
    print("Entropy: ",entropy(dt["Sex"]))
    print("Information gain: ",info_gain(dt["Survived"], dt["Sex"] == "male"))

if __name__ == '__main__':
    main()



























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
