import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(filename):
	df = pd.read_csv(filename, sep=",", index_col=False)
	df.columns = ["housesize", "rooms", "price"]
	data = np.array(df, dtype=float)
	normalize(data)
	return data[:,:2], data[:, -1]

def normalize(data):
	for i in range(0,data.shape[1]-1):
		data[:,i] = ((data[:,i] - np.mean(data[:,i]))/np.std(data[:, i]))

# Parameters
theta = np.array([2, 3])
epochs_num = 100
lr =0.001
# we have our normlaized x & y
x,y = load_data("data.txt")

for _ in range(epochs_num):
    n = x.shape[0]
    y_pred = np.matmul(x, theta)
    grad_cost = np.divide(np.matmul((y_pred - y), x), n)
    theta = theta - (lr)*grad_cost
        
print(theta)

