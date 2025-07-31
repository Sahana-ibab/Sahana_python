import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeRegressor

# loading data as pandas dataframe:
def load_dataset():
    df = pd.read_csv('/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv')

    X= df[["age","BMI", "BP", "blood_sugar", "Gender"]]
    y= df["disease_score"]
    return X, y

#  data split:
# def split_dataset(X,y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42 )
#     return X_train, X_test, y_train, y_test

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
    for train_index, test_index in kf.split(X):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        # declaring model
        tree1 = DecisionTreeRegressor(max_depth=2, max_features=4)
        # fitting the model
        tree1.fit(X_train, y_train)

        # predicting target values for test set:
        y_pred=tree1.predict(X_test)

        # R2-score:
        r2=r2_score(y_test, y_pred)
        print("Fold: ", i)
        print("R2-score: ",r2)
        i+=1
    # calling predefined func--> for visualization :
    plot_fn(tree1, X_train, y_train)

# from sklearn.metrics import confusion_matrix
# print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()
