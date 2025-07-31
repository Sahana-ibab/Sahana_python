# Heart disease prediction

#  Importing all the libraries:
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


#  function to load the data:
# it uses pandas library and loads the data
#  output : data in Dataframe formate
def load_data():
    df = pd.read_csv("Heart.csv")
    # print(df.columns)

    return df

# Function for EDA:
#  Input: Data in df formate
#  Output: statistical analysis and visualization of the data and returns pre-processed data
def EDA_preproccessing_data(df):
    print("EDA: ")
    # Dataset Summary : to overview the data
    print("Dataset Overview:")
    print(df.info())  # Structure and data types
    print("\nSummary Statistics:")
    print(df.describe())  # Statistical overview of numerical features
    print("\nFirst 5 Rows of the Dataset:")
    print(df.head())  # Sample data

    # Missing Values
    print("\nMissing Values per Column:")
    print(df.isnull().sum())  # Check for missing values


    # filling NAN values separately because 'thal' has str datatype and "Ca" is int:
    df[["Thal"]] = df[["Thal"]].fillna("null")
    print(df[["Thal"]])
    print(df.isnull().sum())

    df[["Ca"]] = df[["Ca"]].fillna(0)
    print(df[["Ca"]])
    print(df.isnull().sum())

    print(df.head())

    # plotting scatter plot
    sns.pairplot(df, hue="AHD", palette=["blue", "orange"], height=1.5)

    # plt.show()
    print("scatter plot shows the distribution of different features with ADH!")

    # Histograms for Numerical Features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_features:
        plt.figure(figsize=(8, 6))
        plt.hist(df[col], bins=15, alpha=0.7, color='blue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        # plt.show()

    #  could not scale the data:
    # scaler = StandardScaler()
    # df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df


# Function for splitting dataset:
#  Input: Data in df--X and y
#  Output: Returns Data in 70:30::train:test
def data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    return X_train, X_test, y_train, y_test

# Function for main:
#  to call all individual functions
def main():

    df = load_data()
    df = EDA_preproccessing_data(df)
    # print(df)

    X = df[["Age", "Sex", "ChestPain", "RestBP", "Chol", "Fbs", "RestECG", "MaxHR", "ExAng", "Oldpeak", "Slope", "Ca",
            "Thal"]]
    y = df['AHD']

    # X_train, X_test, y_train, y_test = split_dataset(X, y)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    i = 1
    acc_scores = []
    for train_index, test_index in kf.split(X):
        print("Fold: ", i)
        print("-----training-------\n")
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        # scaler = StandardScaler()

        oh = OneHotEncoder(handle_unknown="ignore")
        X_train = oh.fit_transform(X_train)
        # X_train = scaler.fit_transform(X_train)
        print(X_train)
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)

        model = LogisticRegression()
        model.fit(X_train, y_train)


        X_test = oh.transform(X_test)
        # X_test = scaler.transform(X_test)
        y_test = le.transform(y_test)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy : ", acc,"\n")
        acc_scores.append(acc)
        i=i+1

    mean_r2 = np.mean(acc_scores )
    std_r2 = np.std(acc_scores )
    print("Mean accuracy: ", mean_r2)
    print("Standard deviation: ", std_r2)
    print("Conclusion: \nMean is very less as model per formance is very low, but standard deviation is also less model is not overfitted.")
    print("so, as wide range of data in X, data need to be scaled, but it's giving error!")


if __name__ == '__main__':
    main()