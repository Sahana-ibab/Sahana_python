import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
file_path = "/mnt/data/simulated_data_multiple_linear_regression_for_ML.csv"
df = pd.read_csv(file_path)

# Display basic dataset information
print("Dataset Overview:")
print(df.info())

# Display first few rows
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Visualizing target variable distribution (assuming it's the last column)
target_col = df.columns[-1]  # Change manually if needed
plt.figure(figsize=(6, 4))
sns.histplot(df[target_col], bins=30, kde=True, color="blue")
plt.title(f"Distribution of {target_col}")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Handling missing values (Fill numerical columns with median)
df.fillna(df.median(), inplace=True)

# One-hot encoding for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_features = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))

    # Assign correct column names
    encoded_features.columns = encoder.get_feature_names_out(categorical_cols)
    df = df.drop(columns=categorical_cols).reset_index(drop=True)
    df = pd.concat([df, encoded_features], axis=1)

# Feature Scaling (Standardization)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Display transformed dataset
print("\nTransformed Dataset (First 5 Rows):")
print(df.head())

# Save preprocessed data
df.to_csv("/mnt/data/simulated_data_preprocessed.csv", index=False)
print("\nPreprocessed dataset saved as 'simulated_data_preprocessed.csv'.")
