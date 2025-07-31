from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd



# Load the dataset
df = pd.read_csv("SMILES_only.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]

# Split into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

oh = OneHotEncoder(handle_unknown="ignore")
oh.fit(X_train)
X_train = oh.transform(X_train)


# Train Lasso Regression model
lasso = Lasso(alpha=0.01, max_iter=1000)  # L1 regularization with alpha=0.1
lasso.fit(X_train, y_train)

X_test  = oh.transform(X_test)
# Make predictions
y_pred = lasso.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print("Lasso Coefficients:", lasso.coef_)













