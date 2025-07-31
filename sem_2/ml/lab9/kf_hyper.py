import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeRegressor


# loading data as pandas dataframe:
def load_dataset():
    df = pd.read_csv('/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv')

    X= df[["age","BMI", "BP", "blood_sugar", "Gender"]]
    y= df["disease_score"]
    return X, y

# data split:
def split_dataset(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42 )
    return X_train, X_test, y_train, y_test

# hyper_parameter tuning function,
# takes train ,val-set and model as input and returns best max_depth:
def hyper_parameter( X_train1, X_val, y_train1, y_val, model):
    i_max=0
    temp=0
    for max_d in [ 1, 2, 3, 4 ]:
        model_fit = model(max_depth=max_d)
        model_fit.fit(X_train1, y_train1)
        y_pred = model_fit.predict(X_val)
        R2 = r2_score(y_val, y_pred)
        if R2 > temp:
            i_max = max_d
            temp = R2
    return i_max

#  plotting the tree:
def plot_fn(tree1, X_train, y_train):

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))

    from sklearn import tree
    tree.plot_tree(tree1.fit(X_train, y_train), filled=True)
    plt.show()

def main():
    X, y = load_dataset()

    # X_train, X_test, y_train, y_test = split_dataset(X, y)
    kf = KFold( n_splits=10, shuffle=True, random_state=42 )
    i=1
    R2_scores={}
    for train_index, test_index in kf.split(X):
        print("Fold: ", i)
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        #  val set split:
        X_train1, X_val, y_train1, y_val = split_dataset(X_train, y_train)

        print("----Hyperparameter tuning----")
        model = DecisionTreeRegressor
        i_max = hyper_parameter(X_train1, X_val, y_train1, y_val, model)
        print("Selected max-depth: ", i_max)
        # declaring model
        tree1 = model( max_depth = i_max )
        # fitting the model
        tree1.fit( X_train, y_train )

        # predicting target values for test set:
        y_pred = tree1.predict( X_test )

        # R2-score:
        r2=r2_score(y_test, y_pred)
        print("R2-score: ",r2,"\n")
        R2_scores[i] = r2
        i+=1

    print("Model r2 scores: ", R2_scores)
    std_r2 = np.std(list(R2_scores.values()))
    mean_r2 = np.mean(list(R2_scores.values()))
    print("Standard deviation: ", std_r2)
    print("Mean R2-score: ", mean_r2)




    # # calling predefined func--> for visualization :
    # plot_fn(tree1, X_train, y_train)

# from sklearn.metrics import confusion_matrix
# print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))

if __name__ == '__main__':
    main()
