import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# Load data
def load_data():
    df = pd.read_csv("/home/ibab/datasets/Boston.csv")
    X = df.drop(columns=['medv'])
    y = df['medv']
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return  X_train, X_test, y_train, y_test

def main():
    X, y  = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    xg_reg = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    xg_reg.fit(X_train, y_train)

    y_pred = xg_reg.predict(X_test)
    R2 = r2_score(y_test, y_pred)
    print(f"XG-Boosting Regression R2-score: {R2:.2f}")

if __name__ == '__main__':
    main()