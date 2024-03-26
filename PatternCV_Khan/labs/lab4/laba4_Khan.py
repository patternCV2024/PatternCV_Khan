#1
import numpy as np

x_train_6 = np.array([[7, 12], [42, 7], [37, 27], [30, 33], [34, 23], [38, 40], [18, 30], [9, 46]])
y_train_6 = np.array([1, -1, -1, -1, 1, 1, 1, 1])

mw1_6, ml1_6 = np.mean(x_train_6[y_train_6 == 1], axis=0)
mw_1_6, ml_1_6 = np.mean(x_train_6[y_train_6 == -1], axis=0)

sw1_6, sl1_6 = np.var(x_train_6[y_train_6 == 1], axis=0)
sw_1_6, sl_1_6 = np.var(x_train_6[y_train_6 == -1], axis=0)

print('Середнє: ', mw1_6, ml1_6, mw_1_6, ml_1_6)
print('Дисперсії:', sw1_6, sl1_6, sw_1_6, sl_1_6)

x_6 = [15, 20]  # довжина, ширина жука

a_1_6 = lambda x: -(x[0] - ml_1_6) ** 2 / (2 * sl_1_6) - (x[1] - mw_1_6) ** 2 / (2 * sw_1_6)  # Перший класифікатор
a1_6 = lambda x: -(x[0] - ml1_6) ** 2 / (2 * sl1_6) - (x[1] - mw1_6) ** 2 / (2 * sw1_6)  # Другий класифікатор
y_6 = np.argmax([a_1_6(x_6), a1_6(x_6)])  # Обираємо максимум

print('Номер класу (1 - гусениця, -1 - божа корівка): ', y_6)

#2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Вхідні параметри для першого кластеру
rho1 = 0.85
sigma_x1_squared = 1.3
mu_x1 = [-2, -1]
sigma_y1_squared = 1.0
mu_y1 = [-2, -1]

# Вхідні параметри для другого кластеру (за замовчуванням)
rho2 = 0.7
sigma_x2_squared = 2.0
mu_x2 = [0, 3]
sigma_y2_squared = 2.0
mu_y2 = [0, 3]

# моделювання навчальної вибірки для кожного кластеру
N = 1000
x1 = np.random.multivariate_normal(mu_x1, [[sigma_x1_squared, rho1], [rho1, sigma_y1_squared]], N).T
x2 = np.random.multivariate_normal(mu_x2, [[sigma_x2_squared, rho2], [rho2, sigma_y2_squared]], N).T

# обчислення оцінок середнього та коваріаційних матриць для кожного кластеру
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# модель гауссівського баєсівського класифікатора
Py1, L1 = 0.5, 1  # ймовірності появи класів
Py2, L2 = 1 - Py1, 1  # та величини штрафів невірної класифікації

b = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))

x = np.array([-2, -2])  # вхідний вектор у форматі (x, y)
a = np.argmax([b(x, VV1, mm1, L1, Py1), b(x, VV2, mm2, L2, Py2)])  # класифікатор
print(a)

# виведення графіків
plt.figure(figsize=(4, 4))
plt.title(f"Кореляції: rho1 = {rho1}, rho2 = {rho2}")
plt.scatter(x1[0], x1[1], s=10)
plt.scatter(x2[0], x2[1], s=10)
plt.show()

#3

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Вхідні параметри для першого кластеру
rho1 = -0.85  # Изменен знак коэффициента корреляции
sigma_x1_squared = 1.3
mu_x1 = [-2, -1]
sigma_y1_squared = 1.0
mu_y1 = [-2, -1]

# Вхідні параметри для другого кластеру (за замовчуванням)
rho2 = 0.7
sigma_x2_squared = 2.0
mu_x2 = [0, 3]
sigma_y2_squared = 2.0
mu_y2 = [0, 3]

# моделювання навчальної вибірки для кожного кластеру
N = 1000
x1 = np.random.multivariate_normal(mu_x1, [[sigma_x1_squared, rho1], [rho1, sigma_y1_squared]], N).T
x2 = np.random.multivariate_normal(mu_x2, [[sigma_x2_squared, rho2], [rho2, sigma_y2_squared]], N).T

# обчислення оцінок середнього та коваріаційних матриць для кожного кластеру
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# модель гауссівського баєсівського класифікатора
Py1, L1 = 0.5, 1  # ймовірності появи класів
Py2, L2 = 1 - Py1, 1  # та величини штрафів невірної класифікації

b = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))

x = np.array([-2, -2])  # вхідний вектор у форматі (x, y)
a = np.argmax([b(x, VV1, mm1, L1, Py1), b(x, VV2, mm2, L2, Py2)])  # класифікатор
print(a)

# виведення графіків
plt.figure(figsize=(4, 4))
plt.title(f"Кореляції: rho1 = {rho1}, rho2 = {rho2}")
plt.scatter(x1[0], x1[1], s=10)
plt.scatter(x2[0], x2[1], s=10)
plt.show()

#4

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Вхідні параметри для першого кластеру
rho1 = 0.85
sigma_x1_squared = 1.3
mu_x1 = [-2, -1]
sigma_y1_squared = 1.0
mu_y1 = [-2, -1]

# Вхідні параметри для другого кластеру
rho2 = 0.7
sigma_x2_squared = 2.0
mu_x2 = [0, 3]
sigma_y2_squared = 2.0
mu_y2 = [0, 3]

# Вхідні параметри для третього кластеру
rho3 = -0.5
sigma_x3_squared = 1.5
mu_x3 = [-4, 0]
sigma_y3_squared = 1.5
mu_y3 = [-4, 0]

# моделювання навчальної вибірки для кожного кластеру
N = 1000
x1 = np.random.multivariate_normal(mu_x1, [[sigma_x1_squared, rho1], [rho1, sigma_y1_squared]], N).T
x2 = np.random.multivariate_normal(mu_x2, [[sigma_x2_squared, rho2], [rho2, sigma_y2_squared]], N).T
x3 = np.random.multivariate_normal(mu_x3, [[sigma_x3_squared, rho3], [rho3, sigma_y3_squared]], N).T

# обчислення оцінок середнього та коваріаційних матриць для кожного кластеру
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)
mm3 = np.mean(x3.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x3.T - mm3).T
VV3 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# модель гауссівського баєсівського класифікатора
Py1, L1 = 1/3, 1  # ймовірності появи класів
Py2, L2 = 1/3, 1  
Py3, L3 = 1/3, 1  

b = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))

x = np.array([0, -4])  # вхідний вектор у форматі (x, y)
a = np.argmax([b(x, VV1, mm1, L1, Py1), b(x, VV2, mm2, L2, Py2), b(x, VV3, mm3, L3, Py3)])  # класифікатор
print("Cluster:", a)

# виведення графіків
plt.figure(figsize=(6, 6))
plt.title(f"Кореляції: rho1 = {rho1}, rho2 = {rho2}, rho3 = {rho3}")
plt.scatter(x1[0], x1[1], s=10, label='Cluster 1')
plt.scatter(x2[0], x2[1], s=10, label='Cluster 2')
plt.scatter(x3[0], x3[1], s=10, label='Cluster 3')
plt.legend()
plt.show()
