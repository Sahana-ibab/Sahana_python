import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
def load_data():
    [X,y]=fetch_california_housing(return_X_y=True)
    return X,y
def main():
    X,y=load_data()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)
    X_train_final,X_val,y_train_final,y_val=train_test_split(X_train,y_train,test_size=0.20,random_state=42)
    #scaling
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train_final)
    X_val_scaled=scaler.transform(X_val)
    #imitializing
    r2_max=0
    degrees=[1,2,3,4,5]
    for degree in degrees:
        poly_model=PolynomialFeatures(degree=degree)
        #this adds polynomial interaction terms to the feature set
        X_train_poly=poly_model.fit_transform(X_train_scaled)
        X_val_poly=poly_model.transform(X_val_scaled)
        #fitting the model and training
        model=LinearRegression()
        model.fit(X_train_poly,y_train_final)
        y_pred_poly=model.predict(X_val_poly)
        #evaluation of polynomial
        #r2score
        r2=r2_score(y_val,y_pred_poly)
        print(f"r2 score for degree{degree} in polynomial regression is {r2}")
        if r2>r2_max:
            r2_max=r2
            d=degree
            print(f"r2 score for degree {d} is better")
    print(f"Training the final model after choosing the degree {d} from the val sets using polynomial regression")
    poly_model=PolynomialFeatures(degree=d)
    #retraining the model using training+val set
    #to combine and get the original train set
    X_combined_train=np.vstack((X_train_scaled,X_val_scaled))
    y_combined_train=np.hstack((y_train_final,y_val))
    #transforming for final model
    X_combined_final=poly_model.fit_transform(X_combined_train)
    #scaling for X_test
    X_test_scaled = scaler.transform(X_test)
    #transforming for polynomial model
    X_test_poly = poly_model.transform(X_test_scaled)
    final_model=LinearRegression()
    final_model.fit(X_combined_final,y_combined_train)
    #evaluation
    y_test_pred=final_model.predict(X_test_poly)
    #computing r2 and mse for the test set
    r2_test=r2_score(y_test,y_test_pred)
    mse_test=mean_squared_error(y_test,y_test_pred)
    print(f"r2 score for degree {d} in polynomial regression is {r2_test}")
    print(f"mse for degree {d} in polynomial regression is {mse_test}")
main()








