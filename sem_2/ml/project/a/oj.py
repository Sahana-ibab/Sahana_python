import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score


# Step 1: Load and preprocess the data
def load_preprocess_data():
    data = pd.read_csv("/home/ibab/datasets_IA2/OJ.csv")
    # Step 1: Separate features and target
    X = data.drop(columns='Purchase')
    y = data['Purchase']

    # Step 2: Encode categorical variables in X
    X = pd.get_dummies(X, drop_first=True)

    # Step 3: Encode target
    y = LabelEncoder().fit_transform(y)

    return train_test_split(X, y, train_size=1000, random_state=1)

# Step 2: Train Linear SVM with C=0.01
def train_linear_svm(X_train, y_train, X_test, y_test):
    model = LinearSVC(C=0.01, max_iter=10000, random_state=1)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"(b) Linear SVM (C=0.01) -> Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
    # return model

# Step 3: Use GridSearchCV to find best C
def tune_C(X_train, y_train):
    param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10]}
    grid = GridSearchCV(LinearSVC(max_iter=10000), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print(f"(c) Best C from GridSearch: {grid.best_params_['C']}")
    return grid.best_estimator_

# Step 4: Evaluate model with best C
def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"(d) Tuned Linear SVM -> Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")

# Step 5: Train and test SVM with RBF kernel
def rbf_svm(X_train, y_train, X_test, y_test):
    model = SVC(kernel='rbf', random_state=1)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"(e) RBF SVM -> Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")

# Main function to run all steps
def main():
    X_train, X_test, y_train, y_test = load_preprocess_data()
    train_linear_svm(X_train, y_train, X_test, y_test)
    best_model = tune_C(X_train, y_train)
    evaluate_model(best_model, X_train, y_train, X_test, y_test)
    rbf_svm(X_train, y_train, X_test, y_test)

# Run the main function
if __name__ == '__main__':
    main()
