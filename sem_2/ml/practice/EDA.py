import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  load data
df = pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")
print(df.info)
print(df.describe())

# # Histograms
# df.hist(figsize=(10, 6), bins=15, edgecolor="black")
# plt.suptitle("Feature Distributions", fontsize=14)
# plt.show()
#
# # Ensure Gender is categorical
# df["Gender"] = df["Gender"].map({0: "Male", 1: "Female"})
#
# Create pairplot
sns.pairplot(df, hue="Gender", palette=["blue", "orange"], height=1.5)

plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x=df["age"], y=df["disease_score"], hue=df["Gender"], alpha=0.7)
plt.title("Age vs Disease Score")
plt.xlabel("Age")
plt.ylabel("Disease Score")
# plt.show()

# to see count of null values in respected columns
print(df.isnull().sum())

#  if null has to be filled with 0s:
df = df.fillna(0)
