##1
import numpy as np

def get_polynom(coords):
    n = len(coords)  # Кількість точок
    x_values = [coord[0] for coord in coords]  # Список значень x
    y_values = [coord[1] for coord in coords]  # Список значень y

    # Створення матриці коефіцієнтів системи рівнянь
    a = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a[i, j] = x_values[i] ** j

    # Створення вектора вільних членів
    b = np.array(y_values)

    # Знаходження коефіцієнтів многочлена за допомогою розв'язання системи лінійних рівнянь
    coefficients = np.linalg.solve(a, b)

    return coefficients
