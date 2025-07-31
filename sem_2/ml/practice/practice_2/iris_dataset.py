# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- STEP 1: LOAD DATA & EDA ---

# Load the dataset
file_path = "Iris.csv"
df = pd.read_csv(file_path)

# 1.1 Dataset Summary
print("Dataset Overview:")
print(df.info())  # Structure and data types
print("\nSummary Statistics:")
print(df.describe())  # Statistical overview of numerical features
print("\nFirst 5 Rows of the Dataset:")
print(df.head())  # Preview data

# Dropping unnecessary columns (Id column)
df.drop(columns=['Id'], inplace=True)

# 1.2 Checking for Missing Values
print("\nMissing Values per Column:")
print(df.isnull().sum())  # Check for missing values

# 1.3 Histograms for Numerical Features
numerical_features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for col in numerical_features:
    plt.figure(figsize=(8, 6))
    plt.hist(df[col], bins=15, alpha=0.7, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# 1.4 Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_features].corr()  # Correlation of numerical features
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# --- STEP 2: FEATURE ENCODING & TRANSFORMATION ---

# 2.1 Encoding the Target Variable
# Encoding 'Species' column using LabelEncoder
label_encoder = LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])  # Transform to numerical values
print("\nEncoded Classes (Species):", list(label_encoder.classes_))

# 2.2 Scaling Numerical Features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])  # Standardize features

print("\nFirst 5 Rows After Scaling and Encoding:")
print(df.head())

# --- OUTPUT SUMMARY ---
print("\nPreprocessed Dataset Summary:")
print(df.info())
