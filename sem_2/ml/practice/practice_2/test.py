import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("Heart.csv")

# Drop unnecessary column
df.drop(columns=["Unnamed: 0"], inplace=True)

# 1. Exploratory Data Analysis (EDA)
# Checking for missing values
missing_values = df.isnull().sum()
print("Missing Values in Each Column:\n", missing_values)

# Summary statistics
print("\nSummary Statistics:\n", df.describe())

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Histograms
plt.figure(figsize=(12, 10))
df.hist(figsize=(12, 10), bins=15, edgecolor="black")
plt.suptitle("Feature Distributions")
plt.show()

# 2. Feature Encoding and Data Transformation
# Define categorical and numerical features
categorical_features = ["Sex", "Cp", "RestECG", "ExAng", "Thal"]
numerical_features = ["Age", "RestBP", "Chol", "Fbs", "MaxHR", "Oldpeak", "Slope", "Ca"]

# Preprocessing Pipelines
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Combine Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Apply transformations
processed_data = preprocessor.fit_transform(df)

# Convert processed data back to DataFrame
encoded_columns = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features)
all_columns = np.concatenate([numerical_features, encoded_columns])
df_processed = pd.DataFrame(processed_data, columns=all_columns)

# Display transformed data
print("\nTransformed Data Sample:\n", df_processed.head())
