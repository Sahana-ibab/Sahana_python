from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Load the Diabetes dataset
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
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

    # Train RandomForestClassifier
    rf_reg = RandomForestClassifier(
        n_estimators=100,   # Number of trees in the forest
        max_depth=10,       # Limit tree depth to prevent overfitting
        random_state=42,
        min_samples_split=5,  # Minimum samples to split a node
        min_samples_leaf=2    # Minimum samples per leaf node
    )

    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_test)

    # Model evaluation
    acc = accuracy_score(y_test, y_pred )
    print(f"Random Forest Classifier Accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()
