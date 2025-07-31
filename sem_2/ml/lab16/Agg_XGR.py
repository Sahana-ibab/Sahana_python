# Write a Python program to aggregate  predictions from multiple trees
# to output a final prediction for a regression problem.

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Loading diabetes dataset
def load_data():
    df = pd.read_csv("/home/ibab/datasets/Boston.csv")
    X = df.drop(columns=['medv'])
    y = df['medv']
    return X, y

def data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def bagging(X, y):
    models = []
    n_estimator = 50
    # this is the key difference between bagging and Random forest
    n = X.shape[0]

    for i in range(n_estimator):

        idx = np.random.choice(n, size=n, replace=True)
        bagg_x = X.iloc[idx]
        bagg_y = y.iloc[idx]

        model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
        model.fit(bagg_x, bagg_y)
        models.append(model)

    return models

def Aggregate(models, X_test):
    y_pred_list=[]
    for model in models:
        y_pred = model.predict(X_test)
        y_pred_list.append(y_pred)
    y_f_pred = np.mean(y_pred_list, axis=0)

    return y_f_pred



def main():
    X, y = load_data()


    X_train, X_test, y_train, y_test = data_split(X, y)

    models = bagging(X_train, y_train)
    y_pred = Aggregate(models, X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Bagging XG-Boost Regressor MSE: {mse:.4f}")
    print(f"Bagging XG_Boost Regressor R² Score: {r2:.4f}")


if __name__ == '__main__':
    main()