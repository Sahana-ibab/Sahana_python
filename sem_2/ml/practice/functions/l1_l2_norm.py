import pandas as pd
import numpy as np

def L1_norm(df):
    return df.abs().sum()

def L2_norm(df):
    return np.sqrt((df ** 2).sum())  # this is L2_norm for regularization we don't sqrt

# Example DataFrame
df = pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")
print(df.columns)

X = df[['age', 'BMI', 'BP', 'blood_sugar']]

df = pd.DataFrame(X)

print("L1 Norm:\n", L1_norm(df))
print("\nL2 Norm:\n", L2_norm(df))
