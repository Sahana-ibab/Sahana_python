import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import defaultdict, Counter

# This custom classifier is using the joint probability distribution of feature combinations and classes
class JointProbabilityClassifier:
    def fit(self, X, y):
        # We are initializing dictionaries to store how often each feature combo appears with each class
        self.joint_counts = defaultdict(lambda: defaultdict(int))
        self.class_counts = Counter(y)  # Counting how many times each class appears overall
        self.total = len(y)  # Total number of training samples

        # Now we are going through each (features, label) pair and updating the counts
        for xi, label in zip(X, y):
            key = tuple(xi)  # Turning feature vector into a hashable key like (1, 2)
            self.joint_counts[key][label] += 1  # Storing how often this combo appears with this label

    def predict(self, X):
        y_pred = []
        # For each test sample, we are checking which class has the highest joint probability
        for xi in X:
            key = tuple(xi)
            probs = {}
            for c in self.class_counts:
                # We are estimating P(class | x) ∝ P(x and class) from training data
                probs[c] = self.joint_counts[key][c] / self.total if key in self.joint_counts else 0
            # We are picking the class with the maximum estimated probability
            y_pred.append(max(probs, key=probs.get))
        return y_pred

def main():
    # We are starting by reading the Iris dataset (only using two features + target)
    df = pd.read_csv("Iris.csv")
    df = df[['SepalLengthCm', 'SepalWidthCm', 'Species']]  # Selecting relevant columns

    # Now we are adding small random noise to each feature, to simulate slightly messy real-world data
    np.random.seed(42)  # Seeding for reproducibility
    df['SepalLengthCm'] += np.random.normal(0, 0.2, size=len(df))  # Adding noise to Sepal Length
    df['SepalWidthCm'] += np.random.normal(0, 0.2, size=len(df))   # Adding noise to Sepal Width

    # We are discretizing the continuous features into 4 equal-width bins using sklearn's KBinsDiscretizer
    # This step is necessary because our joint probability model works with discrete values
    discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
    X_binned = discretizer.fit_transform(df[['SepalLengthCm', 'SepalWidthCm']]).astype(int)

    # We are adding the binned features back into the dataframe for easier access
    df['SepalLengthBin'] = X_binned[:, 0]
    df['SepalWidthBin'] = X_binned[:, 1]

    # Now we are separating features and labels
    X = df[['SepalLengthBin', 'SepalWidthBin']].values  # Using only binned versions
    y = df['Species'].values

    # We are splitting the dataset into training and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # We are now training our custom joint probability classifier
    jp_clf = JointProbabilityClassifier()
    jp_clf.fit(X_train, y_train)  # Training on the binned features and labels

    # We are predicting on the test data using the learned joint distributions
    y_pred_jp = jp_clf.predict(X_test)

    # We are calculating how accurate our joint probability predictions are
    acc_jp = accuracy_score(y_test, y_pred_jp)

    # Now we are creating and training a basic Decision Tree with a max depth of 2
    # This gives us a simple, interpretable baseline to compare with
    tree = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree.fit(X_train, y_train)  # Fitting the tree to the same training data

    # We are using the trained tree to predict labels on the test set
    y_pred_tree = tree.predict(X_test)

    # Calculating accuracy of the decision tree
    acc_tree = accuracy_score(y_test, y_pred_tree)

    # Finally, we are printing both accuracies so we can compare the two approaches
    print(f"Joint Probability Classifier Accuracy: {acc_jp:.2f}")
    print(f"Decision Tree (max_depth=2) Accuracy: {acc_tree:.2f}")

if __name__ == '__main__':
    main()
