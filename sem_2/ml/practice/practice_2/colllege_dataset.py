import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "College.csv"
df = pd.read_csv(file_path)

# --- STEP 1: EXPLORATORY DATA ANALYSIS (EDA) ---

# 1.1 Dataset Summary
print("Dataset Overview:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nFirst 5 Rows of the Dataset:")
print(df.head())

# 1.2 Missing Values
print("\nMissing Values per Column:")
print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)  # Replace missing numerical values with column mean

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
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# --- STEP 2: FEATURE ENCODING & TRANSFORMATION ---

# 2.1 Encoding Categorical Variables
# Example binary encoding for "Private" column
df['Private'] = df['Private'].map({'Yes': 1, 'No': 0})

# One-Hot Encoding for other categorical variables (if applicable)
categorical_features = []  # Add any additional nominal categorical columns here
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# 2.2 Scaling Numerical Features
scaler = StandardScaler()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\nFirst 5 Rows After Processing:")
print(df.head())

# --- OUTPUTS ---
print("\nPreprocessed Data Summary:")
print(df.info())
