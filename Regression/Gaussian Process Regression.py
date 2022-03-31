"""
## ------- GPR Overview -------- ##

Gaussian Process (GP) is a powerful supervised machine learning method that is largely 
used in regression settings. This method is desirable in practice since:
    it performs quite well in small data regime;
    it is highly interpretable;
    it automatically estimates the prediction uncertainty.
    
So why estimating prediction uncertainty matters? There are two reasons: 
    first, it enables reliable decision-making by informing us how much we can trust a 
        specific prediction; 
    second, it facilitates active learning, meaning that we could intelligently allocate 
        training data that contribute the most to the model performance.

Despite its importance in practical applications, Gaussian Process does not appear a lot 
in machine learning books and tutorials. 
Partly is because GP theories are filled with advanced statistics and linear algebra and 
are not exactly friendly to newcomers.
 
## ------- Model explanation -------- ##

GP predictions is a distribution, not a number. Such that on a given value of 'x', 
it predicts the distribution of 'y', i.e., μ & σ².

It uses the bayesian approach of, initially difining a random funxtion i.e., prior, then
based on evidence i.e., 'x', it updates it's posterior i.e., the undated function.
Now once we have the function, to get the predicted distribution, we add noice i.e., 'c'
to each sample, then to get the final resilt we average them out.

# --- Terminologies --- #
The “Process” part of its name refers to the fact that GP is a random process.
The “Gaussian” part of its name indicates that GP uses Gaussian distribution.

However, to characterize a random process that contains an infinite number of random 
variables, we will need to upgrade a Gaussian distribution to a Gaussian random process.

# --- Working using μ & σ² --- #
Gaussian random process f(.) is characterized by a 
 mean function μ(x) : K*.1/K.y
 covariance function σ²K(x, x*).
Here, σ² denotes the overall process variance, and K(x, x*) is the correlation function, 
also known as the kernel function. 

When x = x* (i.e., if we the train data remain same), K(x, x)=1 (μ & σ² remains same); 
when x ≠ x*, K(x, x*) represents the correlation between f(x) and f(x*).

For a single location x, f(x) follows a Gaussian distribution, i.e., f(x) ~ N(μ(x), σ²);
For an arbitrary set of locations [x₁, x₂,…, xₙ], f = [f(x₁), f(x₂),…, f(xₙ)] 
follow a multivariate Gaussian distribution

GP model sees the observed labels [y₁, y₂,…, yₙ] of the training data as one random draw 
from the above multivariate Gaussian distribution.
As a result, we are able to train a GP model by inferring the most likely μ(x), σ², and K(x, x*)
that generate the observed labels. This is usually done via maximum likelihood estimation.

# --- Kernal features --- #
In GP modeling settings, a kernel function measures the “similarities” between two different
predictions, i.e., a higher kernel function value implies that the two predictions are more 
similar, and vice versa.
Gaussian kernel
Common kernel functions include the 
    cubic kernel, 
    exponential kernel, 
    Gaussian kernel, and 
    Matérn kernel. 
    
In brief,
    At first we have a prob distribution over function given by Gaussion process & a formula 
    to adapt the present distribution to new point accordingly.
    
    Therefore, now we can combine  both to have a algorithm which calculates & adapts our distribution
    iteratively to the points we observe & we call it GPR.
    
    At end of the iteration, we will have multiple function with all data points, now GP allows
    you to create the mean of these function, that becomes out single optimized function
    &
    It also gives you a range of uncertinity that helps to know & imporve your model.

ref : https://towardsdatascience.com/implement-a-gaussian-process-from-scratch-2a074a470bce

visit : https://github.com/ShuaiGuo16/Gaussian-Process/blob/master/Gaussian_Process.ipynb
Implementation  ↓

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.linalg import cho_solve
from pyDOE import lhs
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

"""  numpy for matrix manipulations.
     matplotlib for data visualization.
     scipy.optimize.minimize: to perform optimization.
     scipy.optimize.Bounds: to specify parameter bounds for optimization.
     pyDOE.lhs: to generate random starting points for the optimizer. 
        Here, we adopt a Latin Hypercube sampling approach because it is good at
        generating evenly distributed random samples.
"""

## Gaussian Process Class :

class GaussianProcess:
    """ A class to contruct the scratch implmentation of
    Gaussian Process Regression 
    
    It will contain 5 major function :
        1. Corr : A correlation function to claculate corr between 
            2 matrix, since we need it as a form of kernels
        2. NegLL : Negative log liklihood function to claculate the 
            -ve log liklihood
        3. ModelFit: A function to train the model
        4. Predict: For predictiong the test data on trained model
        5. Score: To apply MSE and generate score on predictions
    """
    def __init__(self, n_restarts, optimizer):
        """ Initilizer : 
                n_restarts: Number of restart to local optimizer i.e., epochs
                Optimizer: Algo to local optimization      
        """
        self.n_restarts = n_restarts
        self.optimizer = optimizer
    
    def Corr(self, X1, X2, theta):
        """ Construct the Correlation of two 2d Matrix 
            
            Input:
                X1, X2: 2D arrays, shape (n_rows, n_columns)
                Theta: array
            Output:
                K: The correlation matrix.
        """
        # Initilalize K woth Zeros:
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            K[i,:] = np.exp(-np.sum(theta*(X1[i,:]-X2)**2, axis=1))
        return K
    
    def NegLL(self, theta):
        """ To Generate Negative Log Likelihood
            Input:
                Theta: array
            Output:
                - Neg LL
        """
        theta = 10**theta # corr length
        n = self.X.shape[0] # number of rows
        one = np.ones((n,1)) # vector of ones
        
        ## Call Corr
        K = self.Corr(self.X, self.X, theta) + np.eye(n)*1e-10
        L = np.linalg.cholesky(K) # Cholesky matrix decomposition.
        
        ## Estimation of  ---- μ & σ² ---- 
        ## Mean
        mu = (one.T @ (cho_solve((L, True), self.y))) / (one.T @ (cho_solve((L, True), one)))
        ## Variance 
        SigmaSqr = (self.y-mu*one).T @ (cho_solve((L, True), self.y-mu*one)) / n
        
        ## Compute log-likelihood
        LnDetK = 2*np.sum(np.log(np.abs(np.diag(L))))
        LnLike = -(n/2)*np.log(SigmaSqr) - 0.5*LnDetK
        
        ## Update attributes
        self.K, self.L, self.mu, self.SigmaSqr = K, L, mu, SigmaSqr
        
        return -LnLike.flatten()
        
    def modelFit(self, X, Y):
        """ GPR mdoel training
            Input:
                X: Independent Variables
                Y: Dependent Variables
            Output:
                Trained Model
        """
        self.X, self.y = X, Y
        lb, ub = -3, 2
        
        ## Genearte random start point
        lhd = lhs(self.X.shape[1], samples=self.n_restarts)
        
        ## scale random samples to the given bounds
        initial_points = (ub-lb)*lhd + lb
        
        ## Create A Bounds instance for optimization
        bnds = Bounds(lb*np.ones(X.shape[1]),ub*np.ones(X.shape[1]))
        
        ## Run local optimizer on all points
        opt_para = np.zeros((self.n_restarts, self.X.shape[1]))
        opt_func = np.zeros((self.n_restarts, 1))
        for i in range(self.n_restarts):
            res = minimize(self.NegLL, initial_points[i,:], method=self.optimizer,
                bounds=bnds)
            opt_para[i,:] = res.x
            opt_func[i,:] = res.fun
        
        ## Locate the optimum results
        self.theta = opt_para[np.argmin(opt_func)]
        
        ## Update attributes
        self.NegLnlike = self.NegLL(self.theta)
    
    def predict(self, X_test):
        """GP model predicting
        Input:
            Xtest: test set, array of shape (n_samples, n_features)
        
        Output:
            f: GP predictions
            SSqr: Prediction variances"""
        
        n = self.X.shape[0]
        one = np.ones((n,1))
        
        ## Construct correlation matrix between test and train data
        k = self.Corr(self.X, X_test, 10**self.theta)
        
        ## Mean prediction ---- Optimized Mean
        f = self.mu + k.T @ (cho_solve((self.L, True), self.y-self.mu*one))
        
        ## Variance prediction ---- Optimized Variance
        SSqr = self.SigmaSqr*(1 - np.diag(k.T @ (cho_solve((self.L, True), k))))
        
        return f.flatten(), SSqr.flatten()
    
    def score(self, X_test, y_test):
        """Calculate root mean squared error
        Input:
            X_test: test set, array of shape (n_samples, n_features)
            y_test: test labels, array of shape (n_samples, )
        
        Output:
            RMSE: the root mean square error"""
        
        y_pred, SSqr = self.predict(X_test)
        RMSE = np.sqrt(np.mean((y_pred-y_test)**2))
        
        return RMSE
        
        
## TEST -----------------------------------------------------

def Test_1D(X):
    """1D Test Function"""
    
    y = (X*6-2)**2*np.sin(X*12-4)
    
    return y
    
## Training data
X_train = np.array([0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1]).reshape(-1,1)
y_train = Test_1D(X_train)

## Testing data
X_test = np.linspace(0.0, 1, 100).reshape(-1,1)
y_test = Test_1D(X_test)

## GP model training
GP = GaussianProcess(n_restarts=10, optimizer='L-BFGS-B') ## Creating Object
GP.modelFit(X_train, y_train) ## To get Optimized μ, σ² & θ

## GP model predicting
y_pred, y_pred_SSqr = GP.predict(X_test) ## predict using Optimized μ, σ² & θ

print(np.mean(y_pred), np.mean(y_pred_SSqr))