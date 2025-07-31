import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

data =  pd.read_csv('/home/ibab/breast_cancer_row.csv')

# X = data.iloc[:, :-1]
# y =  data.iloc[:, -1]
X = data.iloc[:, :-1].astype(str)
y = data.iloc[:, -1].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=20)
# print(X_train.shape)
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

oe = OrdinalEncoder()
oe.fit(X_train)
X_train = oe.transform(X_train)
# print(X_train.shape)
X_test = oe.transform(X_test)
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc= accuracy_score(y_test, y_pred)
print("Accuracy: ",acc)

from sklearn.metrics import confusion_matrix
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))

from sklearn import tree
tree.plot_tree(model.fit(X_train, y_train), filled=True)
plt.show()
