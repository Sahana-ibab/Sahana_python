import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# --- STEP 1: LOAD DATA & EDA ---

# Load the dataset
file_path = "Hitters.csv"
df = pd.read_csv(file_path)

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
df['Salary'].fillna(df['Salary'].median(), inplace=True)  # Replace NaN in "Salary" column with the median

# 1.3 Histograms for Numerical Features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_features:
    plt.figure(figsize=(8, 6))
    plt.hist(df[col], bins=15, alpha=0.7, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# 1.4 Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# --- STEP 2: FEATURE ENCODING & TRANSFORMATION ---

# 2.1 Feature Encoding
# Encode categorical variables like 'League', 'Division', and 'NewLeague'
df['League'] = df['League'].map({'A': 0, 'N': 1})
df['Division'] = df['Division'].map({'E': 0, 'W': 1})
df['NewLeague'] = df['NewLeague'].map({'A': 0, 'N': 1})

# 2.2 Scaling Numerical Features
scaler = StandardScaler()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\nFirst 5 Rows After Processing:")
print(df.head())

# --- OUTPUT SUMMARY ---
print("\nPreprocessed Dataset Summary:")
print(df.info())
