from sklearn import linear_model
import numpy as np

reg = linear_model.Ridge()
x = np.arange(1e3)
y = 0.5 * x
x = x.reshape(-1, 1) # the parameter x of the fit method should be an 2-d array
reg.fit(x, y)
print(reg.coef_)
# expected output should be:
# [ 0.49999999]
