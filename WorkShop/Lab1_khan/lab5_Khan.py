#1 

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

x_train_6 = np.array([[49, 21], [5, 5], [37, 32], [21, 25], [34, 28], [44, 35], [39, 41], [17, 45], [31, 24]])
y_train_6 = np.array([-1, 1, 1, -1, -1, -1, 1, -1, 1])

# Linear SVM
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(x_train_6, y_train_6)

# Non-linear SVM with Radial Basis Function (RBF) kernel
clf_nonlinear = svm.SVC(kernel='rbf', gamma='auto')
clf_nonlinear.fit(x_train_6, y_train_6)

# Visualizing the results
plt.figure(figsize=(12, 5))

# Plotting linear SVM results
plt.subplot(1, 2, 1)
plt.scatter(x_train_6[:, 0], x_train_6[:, 1], c=y_train_6, cmap=plt.cm.coolwarm, s=30)
plt.scatter(clf_linear.support_vectors_[:, 0], clf_linear.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
plt.title('Linear SVM')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plotting non-linear SVM results
plt.subplot(1, 2, 2)
plt.scatter(x_train_6[:, 0], x_train_6[:, 1], c=y_train_6, cmap=plt.cm.coolwarm, s=30)
plt.scatter(clf_nonlinear.support_vectors_[:, 0], clf_nonlinear.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
plt.title('Non-linear SVM with RBF Kernel')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()


#2

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Linear SVM predictions
linear_predictions = clf_linear.predict(x_test)

# Non-linear SVM with RBF kernel predictions
nonlinear_predictions = clf_nonlinear.predict(x_test)

# Compute metrics for Linear SVM
linear_accuracy = accuracy_score(y_test, linear_predictions)
linear_precision = precision_score(y_test, linear_predictions, zero_division=1) # Add zero_division parameter
linear_recall = recall_score(y_test, linear_predictions, zero_division=1) # Add zero_division parameter
linear_f1 = f1_score(y_test, linear_predictions, zero_division=1) # Add zero_division parameter
linear_confusion_matrix = confusion_matrix(y_test, linear_predictions)

# Compute metrics for Non-linear SVM with RBF kernel
nonlinear_accuracy = accuracy_score(y_test, nonlinear_predictions)
nonlinear_precision = precision_score(y_test, nonlinear_predictions, zero_division=1) # Add zero_division parameter
nonlinear_recall = recall_score(y_test, nonlinear_predictions, zero_division=1) # Add zero_division parameter
nonlinear_f1 = f1_score(y_test, nonlinear_predictions, zero_division=1) # Add zero_division parameter
nonlinear_confusion_matrix = confusion_matrix(y_test, nonlinear_predictions)

# Print the metrics
print("Linear SVM Metrics:")
print("Accuracy:", linear_accuracy)
print("Precision:", linear_precision)
print("Recall:", linear_recall)
print("F1-score:", linear_f1)
print("Confusion Matrix:\n", linear_confusion_matrix)

print("\nNon-linear SVM with RBF Kernel Metrics:")
print("Accuracy:", nonlinear_accuracy)
print("Precision:", nonlinear_precision)
print("Recall:", nonlinear_recall)
print("F1-score:", nonlinear_f1)
print("Confusion Matrix:\n", nonlinear_confusion_matrix)


