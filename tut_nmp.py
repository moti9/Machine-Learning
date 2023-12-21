import numpy as np

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

a = np.array([9, 10])
b = np.array([11, 12])

print(np.dot(x, y), "\n")
print(np.dot(a, b), "\n")
print(np.dot(x, a), "\n")
print(np.dot(y, b), "\n")