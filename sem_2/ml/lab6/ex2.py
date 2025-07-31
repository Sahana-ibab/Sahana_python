from sklearn.model_selection import KFold
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


#loading data file:

df = pd.read_csv('/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv')

X= df[["age","BMI", "BP", "blood_sugar", "Gender"]]
y= df["disease_score"]
# print(X)
# print(y)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    scaler= StandardScaler()
    # scaler= scaler.fit(X_train)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("------training------\n")
    model = LinearRegression()
    model.fit( X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print(y_pred)
    # print(y_test)
    r2 = r2_score(y_test,y_pred)
    print(r2)
    plt.plot(y_test, y_pred)
    # plt.show()
