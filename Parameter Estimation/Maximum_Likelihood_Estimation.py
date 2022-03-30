# Maximum Likelihood Estimation in Python :)
# Which are the best estimating parameters of a probability distribution for my model? 

# Maximum likelihood estimators, when a particular distribution is specified, are considered parametric estimators.

# Maximum Likelihood Estimation, simply known as MLE, is a traditional 
# probabilistic approach that can be applied to data belonging to any distribution,
# i.e., Normal, Poisson, Bernoulli,

# It helps find the most likely-to-occur distribution parameters.

# A likelihood function is simply the joint probability function of the data distribution.
# A maximum likelihood function is the optimized likelihood function employed with most-likely
# parameters.

# Why do we use logs ? 
#       The logarithmic form enables the large product function to be converted 
#       into a summation function.
#       Logs are monotonic transformations, so we’ll simplify our computation but maintain our optimal result.

# For each problem, the users are required to formulate 
#   the model: It defines the forulation for cal Prob i.e., P(Y|XB).
#   &
#   The distribution function: A probability distribution for the target variable

# to arrive at the log-likelihood function. 

# Why is is -ve log likelihood?
#       MLE looks for maximizing the log-likelihood function. Therefore,
#       we supply the negative log likelihood as the input function to the ‘minimize’ method.

## Python Implementation
## Linear Regression (Moodel) on Normal Distribution (The dist for parameters)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

## generating data based on the assumption of normal dist
## independent var
x = np.linspace(-10,30,100)
## Gaussian noise
e = np.random.normal(10,5,100)
## Dependent variable
y = 10+4*x+e ## A probability distribution for the target variable is given Normal dist
df = pd.DataFrame({'x':x, 'y':y})

## UDF Python function that can be iteratively called to determine the -ve log-likelihood value.

## The key idea of formulating this function is that it must contain two elements: 
## the first is the model building equation (here, the simple linear regression). 
## The second is the logarithmic value of the probability density function (here, the log PDF of normal distribution).
## Since we need negative log-likelihood, it is obtained just by negating the log-likelihood.

## Since e is normally distributed y will also be normally distributed
2
 ), our outcome variable yy will also be normally distributed.
def MLE_Norm(parameters):
    const, beta, std_dev = parameters
    pred = const + beta*x ## Linear regrssion, pred
    ## calculate LL
    LL = np.sum(stats.norm.logpdf(y, pred, std_dev))
    neg_LL = -1 * LL
    return neg_LL

## The function can be optimized to find the set of parameters that results in the largest sum likelihood over the training dataset.
mle_model = minimize(MLE_Norm, np.array([2,2,2]), method='L-BFGS-B')

## output
print(mle_model)