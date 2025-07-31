import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# --- STEP 1: LOAD DATA & EDA ---

# Load the dataset
file_path = "Income2.csv"
df = pd.read_csv(file_path, index_col=0)

# 1.1 Dataset Summary
print("Dataset Overview:")
print(df.info())  # Structure and data types
print("\nSummary Statistics:")
print(df.describe())  # Statistical overview of numerical features
print("\nFirst 5 Rows of the Dataset:")
print(df.head())  # Sample data

# 1.2 Missing Values
print("\nMissing Values per Column:")
print(df.isnull().sum())  # Check for missing values
# No missing values identified in this dataset

# 1.3 Histograms for Numerical Features
numerical_features = df.columns
for col in numerical_features:
    plt.figure(figsize=(8, 6))
    plt.hist(df[col], bins=15, alpha=0.7, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# 1.4 Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()  # Compute correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# --- STEP 2: FEATURE SCALING (NORMALIZATION) ---

# Apply StandardScaler to scale the numerical features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("\nFirst 5 Rows After Scaling:")
print(df_scaled.head())

# --- OUTPUT SUMMARY ---
print("\nPreprocessed Dataset Summary:")
print(df_scaled.describe())
