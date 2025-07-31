import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ISLP import load_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv('../datasets/heart.csv') #heart.csv' is in your working directory
print(df.head())
# 2. Preprocess
X = df.drop('output', axis=1)
y = df['output']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Optional: Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# 4. Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Get predicted probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# 6. Manually compute ROC and PR points
thresholds = np.linspace(0, 1, 100)
tpr_list = []
fpr_list = []
precision_list = []
recall_list = []

for thresh in thresholds:
    y_pred = (y_probs >= thresh).astype(int)

    TP = np.sum((y_test == 1) & (y_pred == 1))
    FP = np.sum((y_test == 0) & (y_pred == 1))
    FN = np.sum((y_test == 1) & (y_pred == 0))
    TN = np.sum((y_test == 0) & (y_pred == 0))

    # Avoid division by zero
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 1
    Recall = TPR

    tpr_list.append(TPR)
    fpr_list.append(FPR)
    precision_list.append(Precision)
    recall_list.append(Recall)

# 7. Plot ROC Curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr_list, tpr_list, label='ROC Curve', color='blue')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# 8. Plot Precision-Recall Curve
plt.subplot(1, 2, 2)
plt.plot(recall_list, precision_list, label='PR Curve', color='green')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.show()
