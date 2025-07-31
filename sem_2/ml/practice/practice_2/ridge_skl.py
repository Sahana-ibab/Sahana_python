import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load California Housing Data
X, y = fetch_california_housing(return_X_y=True)

# Handle Missing Values (if any)
# imputer = SimpleImputer(strategy="median")
# X = imputer.fit_transform(X)

# Splitting dataset into Training (70%) and Testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Scaling Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge Regression Model
ridge_model = Ridge(alpha=0.01, max_iter=1000)  # Small alpha to reduce regularization effect
ridge_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = ridge_model.predict(X_test_scaled)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print("\nTheta (Coefficient) values for Ridge Regression:")
print(ridge_model.coef_)
#

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target  # X = features, y = target (disease progression score)

# Split into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Lasso Regression model
lasso = Lasso(alpha=0.1, max_iter=1000)  # L1 regularization with alpha=0.1
lasso.fit(X_train_scaled, y_train)

# Make predictions
y_pred = lasso.predict(X_test_scaled)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print("Lasso Coefficients:", lasso.coef_)

