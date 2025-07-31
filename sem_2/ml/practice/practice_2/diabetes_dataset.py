import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "diabetes_dataset.csv"
df = pd.read_csv(file_path)

# --- STEP 1: EXPLORATORY DATA ANALYSIS (EDA) ---

# 1.1 Dataset Summary
print("Dataset Overview:")
print(df.info())  # Provides data types, null counts, and basic structure
print("\nSummary Statistics:")
print(df.describe())  # Statistical overview for numerical features
print("\nFirst 5 Rows of the Dataset:")
print(df.head())  # Display the first 5 rows of data

# 1.2 Missing Values
print("\nMissing Values per Column:")
print(df.isnull().sum())  # Check for missing values
# Handling missing values (example: Replace zeros in specific columns with NaN and impute)
columns_to_replace_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_to_replace_zero] = df[columns_to_replace_zero].replace(0, np.nan)
df.fillna(df.median(), inplace=True)  # Replace NaN with column medians

# 1.3 Histograms
numerical_features = df.columns[:-1]  # Exclude the Outcome column
for col in numerical_features:
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

# Feature Encoding (Not needed as there are no categorical columns)
# Transformation: Scaling numerical features
scaler = StandardScaler()
numerical_features = df.columns[:-1]  # Exclude Outcome column
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Print the first few rows after scaling
print("\nFirst 5 Rows After Processing:")
print(df.head())

# --- OUTPUT SUMMARY ---
print("\nPreprocessed Dataset Summary:")
print(df.info())
