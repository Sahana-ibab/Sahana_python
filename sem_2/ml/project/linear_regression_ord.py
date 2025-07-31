from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import numpy as np


# Loading Dataset:
def load_data():
    df = pd.read_csv("SMILES_only.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    # print(X.columns)
    # print(y.columns)
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(10, 5))
    # sns.boxplot(x="cell_type", y="A1BG", data=df, palette="tab10")
    # plt.xticks(rotation=45)  # Rotate labels for better readability
    # plt.yscale("log")
    # plt.title("A1BG Expression Across Cell Types")
    # plt.show()

    return X, y

# to split dataset:
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42  )
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
        print("Fold: ", i)
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        oh = OrdinalEncoder()
        oh.fit(X_train)
        X_train = oh.transform(X_train)
        model = LinearRegression()
        model.fit(X_train, y_train)
        oh = OrdinalEncoder()
        oh.fit(X_test)
        X_test = oh.transform(X_test)
        y_pred=model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(r2)
        i+=1
        R2_scores.append(r2)

    mean_r2 = np.mean(R2_scores)
    std_r2 = np.std(R2_scores)
    print("Mean accuracy: ", mean_r2)
    print("Standard deviation: ", std_r2)


if __name__ == '__main__':
    main()

