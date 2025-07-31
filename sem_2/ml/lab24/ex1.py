
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def load_data():
    df = pd.read_csv("spam_sms.csv",encoding='latin-1')
    # print(df.head())
    # print(df.isnull().sum())
    df = df[['v1', 'v2']]
    df["v1"]= df["v1"].map({'ham': 0, 'spam': 1})
    df.columns = ['label', 'message']
    X = df['message']
    y = df['label']

    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def count_vect(X_train, X_test):
    vectorizer = CountVectorizer(stop_words= 'english')  # Remove common stop words
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train_vec, X_test_vec = count_vect(X_train, X_test)


    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy_score: ", acc)


if __name__ == '__main__':
    main()

