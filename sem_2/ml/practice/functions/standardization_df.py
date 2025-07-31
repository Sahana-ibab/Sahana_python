import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Function to standardize data from scratch
def standardize_from_scratch(X):

    mean = np.mean(X, axis=0)  # Mean for each column
    std = np.std(X, axis=0, ddof=0)  # Standard deviation (N, not N-1 for population std)

    standardized_X = (X - mean) / std  # Z-score normalization
    return standardized_X


# Example dataset

df = pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")
print(df.columns)

X = df[['age', 'BMI', 'BP', 'blood_sugar']]

# Convert to NumPy array
X = X.to_numpy()

# Standardize using our function
X_standardized = standardize_from_scratch(X)

# Convert back to DataFrame
df_standardized = pd.DataFrame(X_standardized, columns=['age', 'BMI', 'BP', 'blood_sugar'])
print("\nStandardized Data (from scratch):\n", df_standardized)



# Verify with sklearn StandardScaler
scaler = StandardScaler()
X_sklearn = scaler.fit_transform(X)

df_sklearn = pd.DataFrame(X_sklearn, columns=['age', 'BMI', 'BP', 'blood_sugar'])
print("\nStandardized Data (Scikit-Learn):\n", df_sklearn)
