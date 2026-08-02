import numpy as np

inp = np.array([1, 2, 3, 4])
out = np.array([5, 6, 7])

a = b = c = d = lr = 1

dw = lr * (a * np.outer(out,inp) + b * inp[None, :] + c * out[:, None] + d)
print(dw)
