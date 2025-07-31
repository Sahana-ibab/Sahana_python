import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "Income1.csv"
df = pd.read_csv(file_path)

# --- STEP 1: EXPLORATORY DATA ANALYSIS (EDA) ---

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
# No missing values identified in this dataset, but ensure data integrity.

# 1.3 Histograms for Numerical Features
numerical_features = ['Education', 'Income']  # Explicitly defined numerical columns
for col in numerical_features:
    plt.figure(figsize=(8, 6))
    plt.hist(df[col], bins=15, alpha=0.7, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# 1.4 Correlation Heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()  # Compute correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# --- STEP 2: FEATURE TRANSFORMATION ---

# 2.1 Scaling Numerical Features
scaler = StandardScaler()
df[['Education', 'Income']] = scaler.fit_transform(df[['Education', 'Income']])

print("\nFirst 5 Rows After Scaling:")
print(df.head())

# --- OUTPUT SUMMARY ---
print("\nPreprocessed Dataset Summary:")
print(df.info())
