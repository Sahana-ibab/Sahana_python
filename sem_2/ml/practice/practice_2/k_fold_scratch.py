#K-fold cross validation. Implement for K = 10. Implement from scratch, then, use scikit-learn methods.
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def load_data():
    data=pd.read_csv('/home/ibab/Downloads/data.csv')
    data=data.drop(['id','Unnamed: 32'],axis=1)
    data['diagnosis']=data['diagnosis'].map({'B':0,'M':1})
    X=data.drop(columns=['diagnosis'])
    y=data['diagnosis']
    return X,y
def shuffle_data(X,y):
    indices=np.arange(len(X))
    indices=np.random.permutation(indices)
    return X.iloc[indices],y.iloc[indices]
def create_folds(X,y,k):
    X,y=shuffle_data(X,y)
    folds=np.array_split(np.arange(len(X)),k)
    return folds
def train_test_split(X,y,fold,folds):
    test_index=folds[fold]
    train_index=np.hstack([folds[i] for i in range(len(folds)) if i!=fold])
    return X.iloc[train_index],X.iloc[test_index],y.iloc[train_index],y.iloc[test_index]
def main():
    X,y=load_data()
    k=10
    folds=create_folds(X,y,k)
    model=LogisticRegression(max_iter=10000)
    accuracies=[]
    for i in range(k):
        X_train,X_test,y_train,y_test=train_test_split(X,y,i,folds)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred)
        accuracies.append(accuracy)
        print(f"Fold {i + 1} Accuracy: {accuracy:.3f}")
    print(f"Mean Accuracy: {np.mean(accuracies):.3f}")

main()
