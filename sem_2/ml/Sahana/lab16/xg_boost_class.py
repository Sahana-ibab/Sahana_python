import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def load_data():
    df = pd.read_csv("/home/ibab/datasets/Weekly.csv")
    X = df.drop(columns=['Direction'])
    y = df['Direction']
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return  X_train, X_test, y_train, y_test

def main():
    X, y  = load_data()

    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = split_data(X, y)

    xg_clf = xgb.XGBClassifier(random_state=42)
    xg_clf.fit(X_train, y_train)
    y_pred = xg_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of XG-Boost Classifier: {accuracy}')

if __name__ == '__main__':
    main()