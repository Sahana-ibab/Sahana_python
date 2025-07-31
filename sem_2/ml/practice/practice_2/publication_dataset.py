# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- STEP 1: LOAD DATA & EDA ---

# Load the dataset
file_path = "Publication.csv"
df = pd.read_csv(file_path)

# 1.1 Dataset Summary
print("Dataset Overview:")
print(df.info())  # Overview of columns, data types, and non-null counts
print("\nSummary Statistics:")
print(df.describe())  # Statistical overview of numerical columns
print("\nFirst 5 Rows of the Dataset:")
print(df.head())

# 1.2 Check for Missing Values
print("\nMissing Values per Column:")
print(df.isnull().sum())  # Check if there are any missing values

# 1.3 Histograms for Numerical Features
numerical_features = ['sampsize', 'budget', 'impact', 'time']  # Define numerical columns
for col in numerical_features:
    plt.figure(figsize=(8, 6))
    plt.hist(df[col], bins=15, alpha=0.7, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# 1.4 Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# --- STEP 2: FEATURE ENCODING & TRANSFORMATION ---

# 2.1 Encoding Categorical Variables
# Encode 'mech' column using LabelEncoder
label_encoder = LabelEncoder()
df['mech'] = label_encoder.fit_transform(df['mech'])  # Transform 'mech' column to numerical values
print("\nEncoded 'mech' Classes:", list(label_encoder.classes_))

# 2.2 Scaling Numerical Features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\nFirst 5 Rows After Scaling and Encoding:")
print(df.head())

# --- OUTPUT SUMMARY ---
print("\nPreprocessed Dataset Summary:")
print(df.info())
