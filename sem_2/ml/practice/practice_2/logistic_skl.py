import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('/home/ibab/Downloads/data.csv')
print(data.columns)
data=data.drop(['id','Unnamed: 32'],axis=1)
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
X=data.drop(columns=['diagnosis'])
y=data['diagnosis']

#splitting
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=23)
#scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
#training
model=LogisticRegression(max_iter=1000)
model.fit(X_train_scaled,y_train)
#prediction
y_pred=model.predict(X_test_scaled)
#evaluating
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy of the model:{accuracy}")
y_prob=model.predict_proba(X_test_scaled)[:,1]
std_dev=np.std(y_prob-y_test)
print(f"std dev:{std_dev}")
#for computing z values and visualizing
# #for 1 feature
# X_feature=X_train_scaled[:,7]
# X_feature_sorted=np.sort(X_feature)
# z=model.intercept_+model.coef_[0,0]*X_feature_sorted
#for all features
no_of_features=X_train_scaled.shape[1]
feature_names=X_train.columns
for i in range(no_of_features):
    X_feat_sorted=np.sort(X_train_scaled[:,i])
    z=model.intercept_+model.coef_[0,i]*X_feat_sorted
    sigmoid=1/(1+np.exp(-z))
    plt.plot(X_feat_sorted,sigmoid,label=feature_names[i])
    plt.xlabel(feature_names[i])
    plt.ylabel("sigmoid output")
    plt.show()
# sigmoid=1/(1+np.exp(-z))
# plt.figure(figsize=(7,10))
# plt.plot(X_feature_sorted,sigmoid,label="sigmoid curve",color='red')
# plt.scatter(X_feature,y_train,color='darkgreen',alpha=0.5,label='Data points')
# plt.xlabel('Feature value(std)')
# plt.ylabel('Probability')
# plt.title('sigmoid curve of Logistic regression')
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Predicted probabilities using the logistic regression model
sigmoid = 1 / (1 + np.exp(-np.dot(X_test_scaled, model.coef_[0]) - model.intercept_))

# True labels (y_test)
y_test = y_test.to_numpy().flatten()  # Convert to a flat array if y_test is a pandas Series

# Create the plot
plt.figure(figsize=(10, 6))

# Scatter plot: predicted probabilities vs true labels
plt.scatter(sigmoid, y_test, color='blue', alpha=0.6, label="Data Points")

# Add title and labels
plt.title("Predicted Probabilities vs True Labels")
plt.xlabel("Predicted Probability")
plt.ylabel("True Label (0 = Benign, 1 = Malignant)")
plt.legend()
plt.show()