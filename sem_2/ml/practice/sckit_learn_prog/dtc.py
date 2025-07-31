import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def load_dataset():
    df = pd.read_csv("/home/ibab/datasets/data.csv")
    # print(df.columns)
    X = df[['id', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]
    y= df['diagnosis']
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 4)
    return X_train, X_test, y_train, y_test



def main():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X_train, y_train)
    y_test = le.transform(y_test)
    y_pred = model.predict(X_test)


    print("Acc: ", accuracy_score(y_test, y_pred))



if __name__ == '__main__':
    main()


