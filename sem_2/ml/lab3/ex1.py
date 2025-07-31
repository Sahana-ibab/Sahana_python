import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


#loading data file:

df = pd.read_csv('/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv')

X= df[["age"]]
y= df["disease_score"]
# print(X)
# print(y)
X_train, X_test, y_train , y_test= train_test_split(X, y, test_size = 0.30, random_state=  999 )

scaler= StandardScaler()       # different datas have different range ...so to equate
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
plt.plot(X_test, y_pred)
print(X_test)
plt.show()

# y= df["disease_score_fluct"]
# # print(X)
# # print(y)
# X_train, X_test, y_train , y_test= train_test_split(X, y, test_size = 0.30, random_state=  999 )
#
# scaler= StandardScaler()       # different datas have different range ...so to equate
# scaler= scaler.fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# print("------training------\n")
# model = LinearRegression()
# model.fit( X_train, y_train)
# y_pred = model.predict(X_test)
# r2 = r2_score(y_test,y_pred)
# print(r2)

# df.hist(figsize=(12, 10), bins=30, edgecolor="black")
# plt.subplots_adjust(hspace=0.7, wspace=0.4)
#
# plt.show()


# pd.plotting.scatter_matrix(df[["age", "BMI", "BP", "blood_sugar", "disease_score"]])
# plt.show()


# taking only age:
# X= df[["age"]]
# y= df["disease_score"]
# print(X)
# print(y)
# X_train, X_test, y_train , y_test= train_test_split(X, y, test_size = 0.30, random_state=  999 )
#
# print("------training------\n")
# model= LinearRegression()
# model.fit( X_train, y_train)
# y_pred= model.predict(X_test)
# r2 = r2_score(y_test,y_pred)
# print(r2)
#
# pd.plotting.scatter_matrix(df[["age","disease_score", "disease_score_fluct"]])
#
# plt.show()


# X= df[["age", "BMI", "BP", "blood_sugar", "Gender"]]
# y= df["disease_score"]
# print(X)
# print(y)
# X_train, X_test, y_train , y_test= train_test_split(X, y, test_size = 0.30, random_state=  999 )
#
# print("------training------\n")
# model= LinearRegression()
# model.fit( X_train, y_train)
# y_pred= model.predict(X_test)
# r2 = r2_score(y_test,y_pred)
# print(r2)
#
# pd.plotting.scatter_matrix(df[["age","disease_score", "disease_score_fluct"]])
#
# # plt.show()
#
# print(model.intercept_)
# print(model.coef_)