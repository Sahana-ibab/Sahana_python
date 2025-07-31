import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
file_path = "/mnt/data/OJ.csv"
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

# Visualizing target variable distribution (if applicable)
target_col = 'Purchase'  # Change this if the target column is different
if target_col in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=target_col, palette="viridis")
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
df.to_csv("/mnt/data/OJ_preprocessed.csv", index=False)
print("\nPreprocessed dataset saved as 'OJ_preprocessed.csv'.")
