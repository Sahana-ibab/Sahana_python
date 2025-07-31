from sklearn.linear_model import  LogisticRegression
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


def load_data():
    df = pd.read_csv("/home/ibab/datasets/diabetes_dataset.csv")
    print(df.columns)

    X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
            'BMI', 'DiabetesPedigreeFunction', 'Age']]

    y = df['Outcome']
    # print(y.value_counts())
    return df,X, y

def split_data(X , y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test



def main():
    df,X, y = load_data()

    cat=[c for c in X.columns if X[c].dtype =='object']
    print(cat)
    cat = y.dtype == 'object'
    print(cat)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = LogisticRegression(max_iter=500)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("R2 score: ", accuracy_score(y_test, y_pred))

    # sns.boxplot(x=y, y=X["Glucose"])
    # plt.title("Glucose Levels Across Target Classes")
    # plt.show()
    # sns.countplot(x=y)
    # plt.title("Class Distribution of Target Variable")
    # plt.show()

    sns.pairplot(df, hue="Outcome", palette=["blue", "orange"], height=1.5)

    plt.show()


if __name__ == '__main__':
    main()