import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Load dataset
df=pd.read_csv('/home/ibab/Downloads/archive/Heart.csv',index_col=0)
print(df.columns)
print(df.head())
print(df.dtypes) #gives datatypes of each column
#Dropping unnecessary columns
# df.drop(columns=["Unnamed:0"],inplace=True)

#EDA
#checking for the missing values
missing_val=df.isnull().sum()
print("Missing values in each column:\n",missing_val)

#summary statistics
print(f"Summary Stats:\n {df.describe()}")

#heatmap can be plotted only with numeric values, so
numeric_df=df.select_dtypes(include=['number'])
# Correlation heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.title("Feature Correlation Heatmap")
# plt.show()

#histogram
plt.figure(figsize=(12,10))
df.hist(figsize=(12,10),bins=15,edgecolor="black")
plt.suptitle("Feature distributions")
plt.show()

#detecting outliers
plt.figure(figsize=(12,8))
sns.boxplot(data=numeric_df)
plt.xticks()
plt.title("Boxplot for outlier detection")
plt.show()
