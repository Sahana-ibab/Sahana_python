import pandas as pd

# Load the dataset
file_path = 'breast_cancer_encoding.csv'
columns = ["age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat", "Class"]
df = pd.read_csv(file_path, header=None, names=columns)

# General overview
print("Dataset Overview:")
print(df.info())

# Display first few rows
print("First 5 rows:")
print(df.head())

# Describe numerical and categorical columns
print("Numerical Columns Summary:")
print(df.describe())

print("Categorical Columns Summary:")
print(df.describe(include=['object']))
import matplotlib.pyplot as plt

# Plot histograms for numerical features
numerical_features = ['deg-malig']  # deg-malig is the only numeric column
for col in numerical_features:
    plt.hist(df[col], bins=10, color='blue', alpha=0.7)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
import seaborn as sns
import numpy as np

# Convert ordinal features to numeric for correlation analysis
ordinal_mapping = {
    "age": ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79"],
    "tumor-size": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54"],
    "inv-nodes": ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26"],
}
for col, categories in ordinal_mapping.items():
    df[col] = df[col].apply(lambda x: categories.index(x) if x in categories else -1)

# Calculate and visualize correlations
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
# Apply one-hot encoding for nominal features
nominal_features = ["menopause", "node-caps", "breast", "breast-quad", "irradiat"]
df = pd.get_dummies(df, columns=nominal_features, drop_first=True)

print("One-Hot Encoded Features:")
print(df.head())
# Ordinal Encoding (already applied in EDA for correlation)
# If needed again:
ordinal_mapping = {
    "age": ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79"],
    "tumor-size": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54"],
    "inv-nodes": ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26"],
}
for col, categories in ordinal_mapping.items():
    df[col] = df[col].apply(lambda x: categories.index(x) if x in categories else -1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['deg-malig'] = scaler.fit_transform(df[['deg-malig']])

print("Normalized Numerical Feature (deg-malig):")
print(df['deg-malig'].head())
