import pandas as pd
import numpy as np

data = pd.read_csv("LR.csv", sep=',', header=None)
x = np.hstack((np.ones((len(data[:].values[:, :-1]), 1)), np.array(data[:].values[:, :-1])))
y = np.array(data[3].values).reshape(-1, 1)

def MLR(x, y):
  xnorm = np.matmul(x.T, x)
  ynorm = np.matmul(x.T, y)
  coef = np.matmul(np.linalg.inv(xnorm), ynorm)
  return coef

def predict(x, coef):
  x = [1] + x
  return (coef[1][0] * x[1]) + (coef[2][0] * x[2]) + (coef[3][0] * x[3]) + coef[0][0]

def MLR_cross(x, y, k=5):
  filler = np.arange(len(x))
  np.random.shuffle(filler)
  xs = np.array_split(x[filler], k)
  ys = np.array_split(y[filler], k)
  total = 0
  c = MLR(x, y)
  
  for i in range(k):
    newx, newy = [], []
    for j in range(k):
        if j != i:
            newx.extend(xs[j])
            newy.extend(ys[j])
    newx = np.array(newx)
    newy = np.array(newy)
    c = MLR(newx, newy)
    for j in range(len(xs[i])):
      error = (predict(xs[i][j], c) - ys[i][j]) ** 2
      total += error

  gen = total / len(x)
  point = (predict(x[0], c) - y[0]) ** 2
  return gen, point

c = MLR(x, y)
print("Multiple Linear Regression Model: y =", c[1][0], "x1 +", c[2][0], "x2 +", c[3][0], "x3 +", c[0][0])
print("Predicted value for (1,1,1):", predict([1,1,1], c))
print("Predicted value for (2,0,4):", predict([2,0,4], c))
print("Predicted value for (3,2,1):", predict([3,2,1], c))
gen, point = MLR_cross(x, y)
print("Cross Validation on MLR:", gen[0], point[0])