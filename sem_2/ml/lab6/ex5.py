# Use validation set to do feature and model selection.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt

# X = np.random.uniform(1, 10, 100)
# y = np.random.uniform(1, 10, 100)
data=pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")
X=data[["age", "BMI", "BP", "blood_sugar", "Gender"]]
y=data["disease_score"]

# X=X.reshape(3, 100)
X_temp, X_test ,y_temp, y_test= train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val ,y_train, y_val= train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

r2_scores={}
for i in range(1, 20, 2):
    print("Degree: ", i)
    poly = PolynomialFeatures(degree=i, include_bias=False)

    poly_features= poly.fit_transform(X_train)
    poly_val= poly.transform(X_val)
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_train)
    print("Thetas: ",poly_reg_model.intercept_, poly_reg_model.coef_)

    y_pred = poly_reg_model.predict(poly_val)
    print("y_pred from val set", y_pred)

    r2 = r2_score(y_val,y_pred)
    print("r2 score: ",r2,"\n")

    r2_scores[i]=r2
print(r2_scores,"\n")

poly = PolynomialFeatures(degree=1, include_bias=False)
poly_features = poly.fit_transform(X_train)
poly_val = poly.transform(X_val)
poly_test = poly.transform(X_test)

poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, y_train)

y_t_pred = poly_reg_model.predict(poly_test)
print("y_pred from test set:", y_t_pred)

r2_test = r2_score(y_test, y_t_pred)
print("R² score for test set:", r2_test)

degrees = list(r2_scores.keys())
r2_values = list(r2_scores.values())
plt.plot(degrees, r2_values)
plt.show()
