import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder


def load_data():
    df = pd.read_csv("/home/ibab/datasets_IA2/Weekly.csv")
    X = df[['Lag1', 'Lag2']].values
    y = df['Direction'].values

    y = LabelEncoder().fit_transform(y)

    return X, y


def LOOCV_weekly(X, y):
    n = len(X)
    correct = 0

    for i in range(n):
        X_train = X[np.arange(n) != i]
        y_train = y[np.arange(n) != i]

        X_test = X[i].reshape(1, -1)
        y_test = y[i]

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if y_pred == y_test:
            correct+=1
    acc = correct/n
    print(f"Accuracy of the model: {acc: .3f}")

X, y = load_data()
LOOCV_weekly(X, y)




