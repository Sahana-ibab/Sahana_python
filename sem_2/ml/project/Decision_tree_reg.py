from sklearn.linear_model import LinearRegression, Lasso
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Loading Dataset:
def load_data():
    df = pd.read_csv("SMILES_only_filtered.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    # Standardize Features before Lasso
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply Lasso for Feature Selection
    lasso = Lasso(alpha=0.005, max_iter=5000)
    lasso.fit(X_scaled, y)

    # Select Features
    selected_features = X.columns[lasso.coef_ != 0]
    print(f"Selected Features: {len(selected_features)} out of {X.shape[1]}")

    # Keep only selected features
    X_selected = X[selected_features]

    return X_selected, y, scaler


# to split dataset:
def split_data(X, y, scaler):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42  )
    print("train", X_train.shape)
    print("test", X_test.shape)
    # scaler = StandardScaler()
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def hyper_parameter( X_train1, X_val, y_train1, y_val, model):
    i_max=1
    temp=0
    for max_d in range( 1, 26 ):
        model_fit = DecisionTreeRegressor(max_depth=max_d)
        model_fit.fit(X_train1, y_train1)
        y_pred = model_fit.predict(X_val)
        R2 = r2_score(y_val, y_pred)
        if R2 > temp:
            i_max = max_d
            temp = R2
    return i_max

def main():
    X, y, scaler=load_data()
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    i = 1
    R2_scores = []
    for train_index, test_index in kf.split(X):
        print("Fold: ", i)
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        model = DecisionTreeRegressor()
        X_train1, X_val, y_train1, y_val  = split_data(X_train,y_train, scaler)
        i_max = hyper_parameter(X_train1, X_val, y_train1, y_val, model)
        model = DecisionTreeRegressor(max_depth = i_max)
        model.fit(X_train, y_train)
        y_pred=model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(r2)
        i+=1
        R2_scores.append(r2)

        plt.hist(y_test, alpha=0.5, label="Actual")
        plt.hist(y_pred, alpha=0.5, label="Predicted")
        plt.legend()
        # plt.show()

        # plt.scatter(y_test, y_pred)
        # plt.show()

    mean_r2 = np.mean(R2_scores )
    std_r2 = np.std(R2_scores)
    print("Mean accuracy: ", mean_r2)
    print("Standard deviation: ", std_r2)



if __name__ == '__main__':
    main()

