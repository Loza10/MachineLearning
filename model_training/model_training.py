import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("pima-indians-diabetes.csv", sep=',')
all_y = data["HasDiabetes"].values
all_x = data.drop("HasDiabetes", axis=1).values

def split(data, random_state=None):
  if random_state is None:
    random_state = np.random
    
  main_set = data.values
  np.random.shuffle(main_set)

  train_set = main_set[:500]
  test_set = main_set[500:]
  
  return train_set, test_set

def mm_norm(data):
  mm_min = np.min(data)
  mm_max = np.max(data)

  norm_data = []
  for i in data:
    val = (i - mm_min) / (mm_max - mm_min)
    norm_data.append(val)
  return norm_data

def sigmoid(t):
  n = 1 / (1 + np.exp(-t))
  return n

def SGD_Log(x, y, step):
  weights = np.zeros(x.shape[1])
  for i in range(x.shape[0]):
    index = np.random.randint(x.shape[0])
    xi = x[index]
    yi = y[index]
            
    weights -= step * (sigmoid(np.matmul(xi, weights)) - yi) * xi

  return weights

def SGD_Linear(x, y, step):
  weights = np.zeros(x.shape[1])
  for i in range(x.shape[0]):  
    index = np.random.randint(x.shape[0])
    xi = x[index]
    yi = y[index]
    
    weights -= step * (np.matmul(xi, weights) - yi) * xi
    
  return weights
#-------------------LOSS FUNCTIONS-------------------
def SGD_Log_Loss(x, y, step, epoch):
  weights = np.zeros(x.shape[1])
  losses = []
  
  for j in range(1, epoch):
    for i in range(x.shape[0]):
      index = np.random.randint(x.shape[0])
      xi = x[index]
      yi = y[index]

      if (j % 100 == 0):
        losses.append((predict(xi, weights) - yi)**2)
      weights -= step * (sigmoid(np.matmul(xi, weights)) - yi) * xi
  return weights, losses

def SGD_Linear_Loss(x, y, step, epoch):
  weights = np.zeros(x.shape[1])
  losses = []

  for j in range(1, epoch):
    for i in range(x.shape[0]):  
      index = np.random.randint(x.shape[0])
      xi = x[index]
      yi = y[index]
    if (j % 100 == 0):
      losses.append((np.matmul(xi, weights) - yi) ** 2 / 2)
    weights -= step * (np.matmul(xi, weights) - yi) * xi
  return weights, losses 

def predict(x, w):
    return np.where(sigmoid(np.dot(x, w)) >= 0.5, 1, 0)
  
train, test = split(data)
train = np.array(mm_norm(train))
y_data = train[:, 8]
x_data = np.delete(train, 8, 1)
yt_data = test[:, 8]
xt_data = np.delete(test, 8, 1)
print("Logistic SGD for 0.8: ", SGD_Log(x_data, y_data, 0.8))
print("Logistic SGD for 0.001: ", SGD_Log(x_data, y_data, 0.001))
print("Logistic SGD for 0.000001: ", SGD_Log(x_data, y_data, 0.000001))
print("-----------------------------------------------------")
print("Linear SGD for 0.8: ", SGD_Linear(x_data, y_data, 0.8))
print("Linear SGD for 0.001: ", SGD_Linear(x_data, y_data, 0.001))
print("Linear SGD for 0.000001: ", SGD_Linear(x_data, y_data, 0.000001))


steps = [0.8, 0.001, 0.000001]
fig, axs = plt.subplots(figsize=(8, 14))
for i in steps:
  real_log_loss = []
  count = 0
  lin_weights, lin_loss = SGD_Linear_Loss(x_data, y_data, i, 501)
  log_weights, log_loss = SGD_Log_Loss(x_data, y_data, i, 501)

  for i in range(len(log_loss)):
    count += log_loss[i]
    if (i % 500 == 0):
      real_log_loss.append(count / 500)
      count = 0
  
  axs.plot(range(0, 500, 100), real_log_loss, label="Logistic")
  axs.plot(range(0, 500, 100), lin_loss, label="Linear")
  axs.legend()
  axs.set_xlabel("Epoch")
  axs.set_ylabel("Loss")
  axs.set_title("Loss Comparison 0.000001")
  plt.show()

print("-----------------------------------------------------")
print("Normalized for SGD Logistic on 0.8: ", np.linalg.norm(SGD_Log(x_data, y_data, 0.8)))
print("Normalized for SGD Logistic on 0.001: ", np.linalg.norm(SGD_Log(x_data, y_data, 0.001)))
print("Normalization for SGD Logistic on 0.000001: ", np.linalg.norm(SGD_Log(x_data, y_data, 0.000001)))
print("-----------------------------------------------------")
print("Normalized for SGD Linear on 0.8: ", np.linalg.norm(SGD_Linear(x_data, y_data, 0.8)))
print("Normalized for SGD Linear on 0.001: ", np.linalg.norm(SGD_Linear(x_data, y_data, 0.001)))
print("Normalization for SGD Linear on 0.000001: ", np.linalg.norm(SGD_Linear(x_data, y_data, 0.000001)))

print("-----------------------------------------------------")
step = 0.000001
new_w = SGD_Log(x_data, y_data, step)

def make_predictions(x, y, w, step):
  lowest_error = 0
  best_step = 0
  best_weights = w

  for t in range(1, 100001):
    i = (t - 1) % x.shape[0] 
    xi = x[i]
    yi = y[i]
    w += step * (yi - np.dot(xi, w)) * xi

    if t % 100 == 0:
      y_pred_test = np.dot(x, w)
      sse = np.sum((y - y_pred_test)**2)
      if (lowest_error == 0):
        lowest_error = sse
        best_step = t
        best_weights = w
      elif (lowest_error > sse):
        lowest_error = sse
        best_step = t
        best_weights = w
      print("For Step: ", t, " SSE: ", sse)
      if (np.dot(xi, w) >= 0.5):
        print("For Step: ", t, " Prediction: 1")
      else:
        print("For Step: ", t, " Prediction: 0")
  print("Lowest SSE: ", lowest_error)
  print("Best Weights: ", best_weights)
  return best_step

print("Best Model: ", make_predictions(xt_data, yt_data, new_w, step))
