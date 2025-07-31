from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


# Loading Dataset:
def load_data():
    df = pd.read_parquet("de_train.parquet")
    X = df.iloc[ :, :5 ]
    y = df.iloc[ :, 5 ]
    # print(X.columns)
    # print(y.columns)
    return X, y

# to split dataset:
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42  )
    return X_train, X_test, y_train, y_test


def main():
    X, y=load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)
    oh = OrdinalEncoder()
    oh.fit(X_train)
    X_train = oh.transform(X_train)
    model = LinearRegression()
    model.fit(X_train, y_train)
    oh = OrdinalEncoder()
    oh.fit(X_test)
    X_test = oh.transform(X_test)
    y_pred=model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(r2)

if __name__ == '__main__':
    main()

