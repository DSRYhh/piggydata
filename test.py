import numpy as np

# create numpy array with builtin function
a = np.arange(16)
a = a.reshape((4, 4))  # reshape to a square matrix
print(a)

# choose a part
print(a[1:3, 0:2])

# create numpy array from python list
b = np.array([1, 2, 3, 4])
print(b)

# add with broadcast
print(a + b)

print(np.vstack((a, b)))
