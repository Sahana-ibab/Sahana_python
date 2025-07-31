import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
file_path = "/mnt/data/Wage.csv"
df = pd.read_csv(file_path)

# Display basic dataset information
print("Dataset Overview:\n")
print(df.info())

# Display first few rows
print("\nFirst 5 rows of dataset:")
print(df.head())

# Checking for duplicate rows
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Checking for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])  # Display only columns with missing values

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Distribution of target variable (assuming 'wage' is the target)
target_col = "wage"  # Adjust if needed
plt.figure(figsize=(6, 4))
sns.histplot(df[target_col], bins=30, kde=True, color="blue")
plt.title(f"Distribution of {target_col}")
plt.xlabel(target_col)
plt.show()

# Checking correlation between numerical variables
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Checking relationship of wage with categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], y=df[target_col])
    plt.xticks(rotation=45)
    plt.title(f"{target_col} Distribution by {col}")
    plt.show()

# Handling missing values (Fill numerical columns with median)
df.fillna(df.median(), inplace=True)

# One-hot encoding for categorical columns
if len(categorical_cols) > 0:
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown="ignore")
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
df.to_csv("/mnt/data/Wage_preprocessed.csv", index=False)
print("\nPreprocessed dataset saved as 'Wage_preprocessed.csv'.")
