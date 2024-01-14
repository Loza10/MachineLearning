import pandas as pd
import numpy as np
import math

data = pd.read_csv("LR.csv", sep=',', header=None)

x = data[0].values
y = data[3].values

def SLR(x, y):
  num, den = 0, 0
  mean_x, mean_y = np.mean(x), np.mean(y)
  n = np.size(x)
  for i in range(n):
    num += y[i] * x[i]
    den += x[i] * x[i]
  num -= n * mean_y * mean_x
  den -= n * mean_x * mean_x
  slope = num / den
  yint = mean_y - slope * mean_x
  return slope, yint
  
def predict(x, s, b):
  return b + s * x

def cross_val(x, y, k=5):
  df = np.array([x, y])
  np.random.shuffle(df)
  x = df[0]
  y = df[1]
  
  size = math.ceil(len(x) / k)
  xs, ys = [], []
  total = 0
  s, b = SLR(x, y)
  
  for i in range(0, len(x), size):
    xs.extend([x[i:i+size]])
    ys.extend([y[i:i+size]])
  for i in range(k):
    newx, newy = [], []
    for j in range(k):
        if j != i:
            newx.extend(xs[j])
            newy.extend(ys[j])
    s,b = SLR(newx, newy)
    for j in range(len(xs[i])):
      error = (predict(xs[i][j], s, b) - ys[i][j]) ** 2
      total += error

  gen = total / len(x)
  point = (predict(x[0], s, b) - y[0]) ** 2

  return gen, point

s, b = SLR(x, y)
print("Model for SLR: y =", s, "x +", b)
print("Prediction for 1, 2, 3:", predict(1, s, b), predict(2, s, b), predict(3, s, b))
print("Cross Validation on Linear Model:", cross_val(x, y))
print("---------------------")

polyx = x
z = 1
for z in range(1, 6):
  for i in range(len(x)):
    polyx[i] = x[i]**z
  s, b = SLR(polyx, y)
  print("Model for PR with x: y =", s, "x +", b, "to the", z, "power")
  print("Prediction for 1, 2, 3:", predict(1, s, b), predict(2, s, b), predict(3, s, b))
  print("Cross Validation on polynomial Model:", cross_val(polyx, y))
  print("---------------------")
