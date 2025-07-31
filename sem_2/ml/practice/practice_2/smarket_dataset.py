# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- STEP 1: LOAD DATA & EDA ---

# Load the dataset
file_path = "Smarket.csv"
df = pd.read_csv(file_path)

# 1.1 Dataset Summary
print("Dataset Overview:")
print(df.info())  # Overview of columns, data types, and null values
print("\nSummary Statistics:")
print(df.describe())  # Statistical summary of numerical features
print("\nFirst 5 Rows of the Dataset:")
print(df.head())  # Display first few rows

# 1.2 Check for Missing Values
print("\nMissing Values per Column:")
print(df.isnull().sum())  # Check for any missing values

# 1.3 Histograms for Numerical Features
numerical_features = ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today']
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
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# --- STEP 2: FEATURE ENCODING & TRANSFORMATION ---

# 2.1 Encode the Categorical Target Variable
# Encode the 'Direction' column using LabelEncoder ('Up' -> 1, 'Down' -> 0)
label_encoder = LabelEncoder()
df['Direction'] = label_encoder.fit_transform(df['Direction'])
print("\nEncoded Classes (Direction):", list(label_encoder.classes_))

# 2.2 Scaling Numerical Features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Display the first few rows after processing
print("\nFirst 5 Rows After Scaling and Encoding:")
print(df.head())

# --- OUTPUT SUMMARY ---
print("\nPreprocessed Dataset Summary:")
print(df.info())
