from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np




# Loading diabetes dataset
def load_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    return X, y


def data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def bagging(X, y):
    models = []
    n_estimator = 50
    # this is the key difference between bagging and Random forest
    n = int(np.sqrt(np.shape(X)[0]))

    for i in range(n_estimator):
        idx = []
        for j in range(n):
            idx.append(np.random.choice(n, replace=True))

        bagg_x = X[idx]
        bagg_y = y[idx]

        model = DecisionTreeClassifier()
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

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = bagging(X_train, y_train)
    y_pred = Aggregate(models, X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Bagging Regressor MSE: {mse:.4f}")
    print(f"Bagging Regressor R² Score: {r2:.4f}")


if __name__ == '__main__':
    main()