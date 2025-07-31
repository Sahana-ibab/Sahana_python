# Implement Gradient Boost Regression and Classification using
# scikit-learn. Use the Boston housing dataset from the ISLP
# package for the regression problem and weekly dataset from
# the ISLP package and use Direction as the target variable
# for the classification.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder


def load_data():
    df = pd.read_csv("/home/ibab/datasets/Boston.csv")
    X = df.drop(columns=['medv'])  # Features
    y = df['medv']  # Target (medv)
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return  X_train, X_test, y_train, y_test

def main():
    X, y  = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)

    R2 = r2_score(y_test, y_pred)
    print(f"Gradient Boosting Regression R2-score: {R2:.2f}")


if __name__ == '__main__':
    main()