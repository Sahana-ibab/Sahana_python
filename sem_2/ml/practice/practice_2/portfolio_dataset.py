# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# --- STEP 1: LOAD DATA & EDA ---

# Load the dataset
file_path = "Portfolio.csv"
df = pd.read_csv(file_path)

# 1.1 Dataset Summary
print("Dataset Overview:")
print(df.info())  # Overview of column data types and missing values
print("\nSummary Statistics:")
print(df.describe())  # Summary statistics for numerical features
print("\nFirst 5 Rows of the Dataset:")
print(df.head())  # Display first 5 rows

# 1.2 Checking for Missing Values
print("\nMissing Values per Column:")
print(df.isnull().sum())  # Check for any missing values

# 1.3 Visualizing Data Distribution with Histograms
for col in df.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(df[col], bins=15, alpha=0.7, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# 1.4 Scatter Plot of X vs. Y
plt.figure(figsize=(8, 6))
plt.scatter(df['X'], df['Y'], color='green', alpha=0.7)
plt.title('Scatter Plot of X vs Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# 1.5 Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()  # Compute correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# --- STEP 2: FEATURE TRANSFORMATION ---

# Scaling numerical features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("\nFirst 5 Rows After Scaling:")
print(df_scaled.head())

# --- OUTPUT SUMMARY ---
print("\nPreprocessed Dataset Summary:")
print(df_scaled.describe())
