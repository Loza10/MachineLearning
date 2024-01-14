import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_table("crime-train.txt")
test = pd.read_table("crime-test.txt")

y_test = test["ViolentCrimesPerPop"].values
x_test = test.drop("ViolentCrimesPerPop", axis=1).values
y_train = train["ViolentCrimesPerPop"].values
x_train = train.drop("ViolentCrimesPerPop", axis=1).values

lam = 600

def soft(c_a, lam_a):
  if c_a < (-lam_a / 2):
    return (c_a + lam_a / 2)
  elif c_a > lam_a / 2:
    return (c_a - lam_a / 2)
  else:
    return 0
    
def lasso(x, y, lamb):
  N, D = x.shape
  w = np.matmul(np.linalg.inv(np.add(np.matmul(x.T, x), (np.identity(D) * lam))), np.matmul(x.T, y))
  a = 0
  c = 0
  coef = 0
  converge = False
  while True:
    if (converge == True):
      break
    for d in range(D):
      for n in range(N):
        a += x[n, d] * x[n, d]
      for n in range(N):
        dot_product = y[n] - np.matmul(w.T, x[n, :]) + w[d] * x[n, d]
        c += x[n, d] * (dot_product)
      w_temp = soft(c / a, lamb / a)
      if abs(w[d] - w_temp) < (10 ** -6):
        converge = True
      w[d] = w_temp
      if (w[d] > 0):
        coef += w[d]
    pred = np.matmul(x,w)
    error = np.mean((y-pred)**2)
  return w, error, coef

weights = []
lambval = []
c = []
train_error = []
test_error = []
temp = 0
nonzero = 0
known = False

while True:
  lambval.append(lam)
  new_w, temp, co = lasso(x_train, y_train, lam)
  lam = lam / 2
  weights.append([new_w[3], new_w[12], new_w[39], new_w[45], new_w[56]])
  train_error.append(temp)
  c.append(co)
  nonzero = np.count_nonzero(new_w)
  if (nonzero >= 95):
    break
  
weights = np.array(weights)
N, D = weights.shape
for i in range(D):
  plt.plot(np.log10(lambval), weights[:, i])
plt.xlabel('Lambda values')
plt.ylabel('Weights')
plt.title('Regularization Path')
plt.show()

plt.plot(np.log10(lambval), train_error)
plt.xlabel('Lambda values')
plt.ylabel('Error (Train)')
plt.title('Lambda vs Error')
plt.show()

for i in range(10):
  new_w, temp, co = lasso(x_test, y_test, lam)
  test_error.append(temp)

plt.plot(np.log10(lambval), test_error)
plt.xlabel('Lambda values')
plt.ylabel('Error (Test)')
plt.title('Lambda vs Error')
plt.show()

plt.plot(lambval, c)
plt.xlabel('Lambda values')
plt.ylabel('Nonzero Coefficients')
plt.title('Lambda vs Coef')
plt.show()