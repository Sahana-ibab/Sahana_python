from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Loading diabetes dataset
def load_data():
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    return X, y


def data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = data_split(X, y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Bagging Regressor with Decision Trees
    bagging_reg = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        n_estimators=50,
        random_state=42
    )

    bagging_reg.fit(X_train, y_train)

    y_pred = bagging_reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Bagging Regressor MSE: {mse:.4f}")
    print(f"Bagging Regressor R² Score: {r2:.4f}")


if __name__ == '__main__':
    main()