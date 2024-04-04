#1

import numpy as np

coefficients = np.array([[0.05, 0.07, 0],
                         [0.05, 0,    0.06],
                         [1,    1,    1]])

results = np.array([2250, 1400, 50000])

solution = np.linalg.solve(coefficients, results)

print("Сума на першому рахунку:", round(solution[0], 2), "умовних одиниць")
print("Сума на другому рахунку:", round(solution[1], 2), "умовних одиниць")
print("Сума на третьому рахунку:", round(solution[2], 2), "умовних одиниць")


#2

import numpy as np

coefficients = np.array([[1, 1, 1], [1, -1, 0], [1, 0, -1]])
constants = np.array([1328, -120, 100])
solution = np.linalg.solve(coefficients, constants)
solution = np.round(solution).astype(int)

print("Кількість iPhone 6:", solution[0])
print("Кількість iPhone 11:", solution[1])
print("Кількість iPhone 12:", solution[2])


#3

import numpy as np

A = np.array([np.sqrt(3), 0, np.sqrt(3)])
B = np.array([np.sqrt(6), 1/2, 0])
C = np.array([1, 1/np.sqrt(3), 1])

coefficients_matrix = np.array([[A[0]**2, 0, A[2]**2],
                                [B[0]**2, (1/2)**2, 0],
                                [1, (1/np.sqrt(3))**2, 1]])

constants_vector = np.array([1, 1, 1])

solution = np.linalg.solve(coefficients_matrix, constants_vector)

print("a^2 =", solution[0])
print("b^2 =", solution[1])
print("c^2 =", solution[2])


#4

import numpy as np

points = np.array([[1, 1, 12],
                   [3, 1, 54],
                   [-1, 1, 2]])

x = points[:, 0]
y = points[:, 2]

A = np.vstack([x**2, x, np.ones(len(x))]).T

a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]

print("Коефіцієнти рівняння параболи:")
print("a =", a)
print("b =", b)
print("c =", c)


#5

import numpy as np

def get_polynom(coords):
    n = len(coords) - 1  

    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    for i in range(n + 1):
        x, y = coords[i]
        b[i] = y
        for j in range(n + 1):
            A[i, j] = x ** j

    coeffs = np.linalg.solve(A, b)
    
    return coeffs

coords = [(1, 2), (2, 3), (3, 5), (4, 7)]
coefficients = get_polynom(coords)
print("Коефіцієнти многочлена:", coefficients)
