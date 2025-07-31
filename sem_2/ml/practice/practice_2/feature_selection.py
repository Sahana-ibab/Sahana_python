import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load Diabetes dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Lasso with cross-validation to find the best alpha
lasso_cv = LassoCV(alphas=np.logspace(-3, 2, 100), cv=5, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

# Get selected features (non-zero coefficients)
selected_features = X.columns[lasso_cv.coef_ != 0]
removed_features = X.columns[lasso_cv.coef_ == 0]

print(f"Best Alpha: {lasso_cv.alpha_:.4f}")
print("Selected Features:", list(selected_features))
print("Removed Features:", list(removed_features))
