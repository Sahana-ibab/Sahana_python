import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def load_dataset():
    df = pd.read_csv("/home/ibab/datasets/simulated_data_multiple_linear_regression_for_ML.csv")
    # print(df.columns)
    X = df[['age', 'BMI', 'BP', 'blood_sugar', 'Gender']]
    y= df['disease_score']
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 4)
    return X_train, X_test, y_train, y_test

def hyper_parameter(X_train1, X_val, y_train1, y_val):
    temp= 0
    i_max = 0
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        model = DecisionTreeRegressor(max_depth=i)
        model.fit(X_train1, y_train1)
        y_v_pred = model.predict(X_val)
        R2 = r2_score(y_val, y_v_pred)
        # print(f"max_depth={i}, R²={R2:.4f}")
        if temp < R2:
            temp = R2
            i_max = i
    return i_max

def main():
    X, y = load_dataset()
    # X_train, X_test, y_train, y_test = split_data(X, y)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    i=0
    r2s=[]
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train1 ,X_val, y_train1, y_val = split_data(X_train, y_train)

        i_max = hyper_parameter(X_train1 ,X_val, y_train1, y_val)
        print(i_max)
        model = DecisionTreeRegressor(max_depth=i_max)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        print("R2_Score: ", r2_score(y_test, y_pred))
        r2s.append(r2)

    mean_r2 = sum(r2s)/len(r2s)
    print("Mean r2 score: ",mean_r2)

if __name__ == '__main__':
    main()


