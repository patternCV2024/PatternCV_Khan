import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Вхідні дані
x_train_6 = np.array([[28, 31],
                      [50, 32],
                      [17, 36],
                      [34, 34],
                      [28, 6],
                      [31, 43],
                      [7, 8],
                      [27, 25],
                      [34, 34],
                      [22, 42]])
y_train_6 = np.array([-1, 1, -1, 1, -1, 1, -1, -1, 1, 1])  # -1 для синіх, 1 для червоних

# Побудова моделі
model = SVC(kernel='linear')
model.fit(x_train_6, y_train_6)

# Класифікація
y_pred = model.predict(x_train_6)
accuracy = accuracy_score(y_train_6, y_pred)
print("Accuracy:", accuracy)

# Відображення даних та розділної границі
plt.figure(figsize=(8, 6))

# Сині точки
plt.scatter(x_train_6[y_train_6 == -1][:, 0], x_train_6[y_train_6 == -1][:, 1], color='blue', label='-1')
# Червоні точки
plt.scatter(x_train_6[y_train_6 == 1][:, 0], x_train_6[y_train_6 == 1][:, 1], color='red', label='1')

# Створення сітки для візуалізації розділної границі
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Створення сітки для прогнозування розділної границі
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Візуалізація розділної границі та меж класів
plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
plt.title('Binary Classification with SVM')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.show()
