from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

df = pd.read_csv("SMILES_only.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]

# Splitting dataset into Training (70%) and Testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# # Scaling Features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


oh = OneHotEncoder(handle_unknown="ignore")
oh.fit(X_train)
X_train = oh.transform(X_train)

# Train Ridge Regression Model
ridge_model = Ridge(alpha=0.01, max_iter=1000)  # Small alpha to reduce regularization effect
ridge_model.fit(X_train, y_train)


# Predictions
X_test = oh.transform(X_test)
y_pred = ridge_model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print("\nTheta (Coefficient) values for Ridge Regression:")
# print(ridge_model.coef_)
#