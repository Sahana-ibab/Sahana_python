import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC


def load_preprocess_data():
    df=pd.read_csv("/home/ibab/datasets_IA2/OJ.csv")

    X = df.drop(columns = 'Purchase')
    y = df['Purchase']

    X = pd.get_dummies(X, drop_first=True)
    y=LabelEncoder().fit_transform(y)

    return train_test_split(X, y, train_size=1000, random_state=1)


def train_linear_svm(X_train, X_test, y_train, y_test):
    model = LinearSVC(C=0.01, max_iter=10000, random_state=1)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Linear SVM: train set acc: {train_acc:.3f}, test set acc {test_acc:.3f}")


def tune_C (X_train, y_train):
    param_grid = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    grid = GridSearchCV(LinearSVC(max_iter=10000), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print(f"Best C for Linear SVM: {grid.best_params_['C']}")
    return grid.best_estimator_

def evaluation_model(model, X_train, X_test, y_train, y_test):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print("Train Acc: ",train_acc,", test acc:" ,test_acc)


def RBF_kernel(X_train, X_test, y_train, y_test):
    model = SVC(kernel='rbf', random_state=1)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"RBF SVM -> Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")


# Main function to run all steps
def main():
    X_train, X_test, y_train, y_test = load_preprocess_data()
    train_linear_svm(X_train, X_test, y_train, y_test)
    best_model = tune_C(X_train, y_train)
    evaluation_model(best_model, X_train, X_test, y_train, y_test)
    RBF_kernel(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()