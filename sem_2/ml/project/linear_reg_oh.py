from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# Loading Dataset:
def load_data():
    df = pd.read_csv("SMILES_only_filtered.csv")
    X = df.iloc[ :, :-1]
    y = df.iloc[ :, -1: ]
    print(df.describe())

    # print(y)
    # print(X)
    # # print(y.head())
    # # plotting scatter plot
    # sns.pairplot(dff ,hue="cell_type", palette="tab10", height=1.5)

    # plt.show()
    return X, y

# to split dataset:
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42  )
    print("train", X_train.shape)
    print("test", X_test.shape)
    return X_train, X_test, y_train, y_test


def main():
    X, y=load_data()
    # X_train, X_test, y_train, y_test = split_data(X, y)
    # print("X_train shape:", X_train.shape)
    # print("X_test shape:", X_test.shape)
    # X_train, X_test, y_train, y_test = split_dataset(X, y)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    i = 1
    R2_scores = []
    for train_index, test_index in kf.split(X):
        print("\nFold: ", i)
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # oh = OneHotEncoder(handle_unknown="ignore")
        # oh.fit(X_train)
        # X_train = oh.transform(X_train)
        # print("train", X_train.shape)
        # print("test", X_test.shape)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        # X_test = oh.transform(X_test)
        print("train", X_train_scaled.shape)
        print("test", X_test_scaled.shape)
        y_pred=model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        print(r2)
        i+=1
        R2_scores.append(r2)

    mean_r2 = np.mean(R2_scores )
    std_r2 = np.std(R2_scores)
    print("Mean accuracy: ", mean_r2)
    print("Standard deviation: ", std_r2)



if __name__ == '__main__':
    main()

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# # Load Data
# df = pd.read_csv("SMILES_only.csv")
# X = df.iloc[ :, :-1]
# y = df.iloc[ :, -1: ]
# # Compute Correlation (features vs. target)
# correlation = df.corr().iloc[:, -1].sort_values(ascending=False)  # Replace 'target_variable' with your y column name
#
# # Plot Correlation
# plt.figure(figsize=(10, 5))
# sns.barplot(x=correlation.index, y=correlation.values)
# plt.xticks(rotation=90)
# plt.title("Feature-Target Correlation")
# plt.ylabel("Correlation Coefficient")
# plt.show()
#
# # Print Correlation Values
# print(correlation)
