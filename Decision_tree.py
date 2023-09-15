import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.255, random_state=8)
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Training the Decision Tree Classification model on the Training set
classifier = DecisionTreeClassifier(criterion='entropy', random_state=5)
classifier.fit(X_train, y_train)
# Display the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(classifier, filled=True, rounded=True, feature_names=data.feature_names[:-1])
plt.show()
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Display the results (confusion matrix and accuracy)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
