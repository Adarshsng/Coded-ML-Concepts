"""
Linear regression is the most basic type of regression commonly used for
predictive analysis. The idea is pretty simple: we have a dataset and we have
features associated with it. Features should be chosen very cautiously
as they determine how much our model will be able to make future predictions.
We try to set the weight of these features, over many iterations, so that they best
fit our dataset. In this particular code, We try to best fit a line through dataset 
and estimate the parameters.
"""

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
theta = np.array([2, 3]) # model parameters randomly initialised 
epochs_num = 100 # epochs
lr =0.001 # learning rate
# we have our normlaized x & y
x,y = load_data("data.txt")

for _ in range(epochs_num): # running in epochs
    n = x.shape[0] # data length 'n'
    y_pred = np.matmul(x, theta) # Hypothesis calculation 
    grad_cost = np.divide(np.matmul((y_pred - y), x), n) # Derivated of Cost
    theta = theta - (lr)*grad_cost # Updating models paramters

# Updated Model Parameter     
print(theta)

