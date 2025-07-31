import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Function to normalize data from scratch
def normalize_from_scratch(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_norm = (X - X_min) / (X_max - X_min)
    return X_norm

# Example dataset
df = pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")
print(df.columns)

X = df[['age', 'BMI', 'BP', 'blood_sugar']]
# Convert to NumPy array
X = X.to_numpy()

# Apply normalization from scratch
X_normalized = normalize_from_scratch(X)

# Convert back to DataFrame
df_normalized = pd.DataFrame(X_normalized, columns=['age', 'BMI', 'BP', 'blood_sugar'])
print("\nNormalized Data (from scratch):\n", df_normalized)

# Verify with sklearn MinMaxScaler
scaler = MinMaxScaler()
X_sklearn = scaler.fit_transform(X)

df_sklearn = pd.DataFrame(X_sklearn, columns=['age', 'BMI', 'BP', 'blood_sugar'])
print("\nNormalized Data (Scikit-Learn):\n", df_sklearn)
