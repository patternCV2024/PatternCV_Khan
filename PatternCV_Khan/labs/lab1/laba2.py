import numpy as np
import matplotlib.pyplot as plt

# Дані
x_train_6 = np.array([[19, 16],
                      [5, 41],
                      [46, 26],
                      [47, 19],
                      [5, 26],
                      [35, 39],
                      [18, 21],
                      [6, 48],
                      [34, 22]])

y_train_6 = np.array([1, -1, 1, 1, 1, -1, -1, -1, 1])

# Мітки класів
y_train_6_new = np.where(y_train_6 == 1, -1, 1)

# Додаємо зміщення для кожного прикладу
x_train_6_bias = np.c_[x_train_6, np.ones(len(x_train_6))]

# Розрахунок коефіцієнтів
pt = np.sum([x * y for x, y in zip(x_train_6_bias, y_train_6_new)], axis=0)  # Обчислення підсумку
xxt = np.sum([np.outer(x, x) for x in x_train_6_bias], axis=0)  # Обчислення підсумку зовнішнього добутку
w = np.dot(pt, np.linalg.inv(xxt))  # Обчислення вагових коефіцієнтів

# Формування координат для лінії розділення
line_x = np.linspace(min(x_train_6[:, 0]), max(x_train_6[:, 0]), 100)
line_y = -w[0]/w[1] * line_x - w[2]/w[1]

# Формування точок для класу 1 та класу -1
x_minus_1 = x_train_6[y_train_6 == -1]
x_1 = x_train_6[y_train_6 == 1]

# Відображення графіку
plt.figure(figsize=(8, 6))
plt.scatter(x_minus_1[:, 0], x_minus_1[:, 1], color='blue', label='Клас -1')
plt.scatter(x_1[:, 0], x_1[:, 1], color='red', label='Клас 1')
plt.plot(line_x, line_y, color='green', label='Лінія розділення')

plt.xlabel("Ознака 1")
plt.ylabel("Ознака 2")
plt.title("Бінарний МНК-класифікатор")
plt.legend()
plt.grid(True)
plt.show()
