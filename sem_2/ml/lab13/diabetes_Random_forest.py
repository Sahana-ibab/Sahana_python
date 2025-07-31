from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the Diabetes dataset
def load_data():
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    return X, y

# Split the dataset into training and testing sets
def data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = data_split(X, y)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train RandomForestRegressor
    rf_reg = RandomForestRegressor(
        n_estimators=100,   # Number of trees in the forest
        max_depth=10,       # Limit tree depth to prevent overfitting
        random_state=42,
        min_samples_split=5,  # Minimum samples to split a node
        min_samples_leaf=2    # Minimum samples per leaf node
    )

    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_test)

    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest Regressor MSE: {mse:.4f}")
    print(f"Random Forest Regressor R² Score: {r2:.4f}")

if __name__ == '__main__':
    main()
