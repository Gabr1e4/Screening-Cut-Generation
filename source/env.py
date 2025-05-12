# import packages
import numpy as np
import pandas as pd
from typing import Union


# ===== Synthetic Data Generation =====
def sparse_generation(
        num:int,
        dim:int,
        sparsity:int = 10,
        rho:float = 0.5,
        SNR:float = 3.5,
        random_seed = None):
    """ The function returns model, i.e. y_i = x_i @ beta_true + epsilon_i, 
        where both x_i and epsilon_i follows a Gaussian distribution, and w has a sparse structure with {0,1,-1} entries
    Args:
        num(int): number of samples
        dim(int): number of features
        sparsity(int): the sparsity level of the ground-truth beta_true
        rho(float): the correlation coefficient
        SNR(float): the Signal to Noise Ratio
        random_seed: set the random seed
    Returns:
        (X,Y,beta_true): The corresponding covariate matrix, response vector and the ground-truth beta_true
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    true_support = np.random.randint(0,dim,sparsity)
    beta_true = np.zeros(dim)
    beta_true[true_support] = 2*np.random.binomial(1,0.5,sparsity)-1
    
    covariance = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            covariance[i,j] = rho**(max(i-j,j-i))

    X = np.random.multivariate_normal(np.zeros(dim),covariance,num)

    # SNR setting
    noise_variance = (np.linalg.norm(X@beta_true,2)**2 / (SNR**2)) / num
    epsilon = np.random.normal(0,noise_variance,num)
    
    Y = X@beta_true + epsilon
    return (X,Y,beta_true)



# ===== Gamma (the coefficient for l2 penalty) Generation =====
def generate_gamma(X,sparsity,i):
    """ The function returns gamma, the coefficient for l2 penalty term
    Args:
        X(np.ndarray): sample matrix
        sparsity(int): the sparsity level of the regression problem
        i(float): the coefficient to generate gamma-zero
    Returns:
        gamma: The corresponding coefficient for l2 penalty term
    """
    num = X.shape[0]
    dim = X.shape[1]
    max_row = max([np.linalg.norm(X[row,:],2)**2 for row in range(num)])
    gamma_zero = dim/(num*sparsity*max_row)
    return (2**i)*gamma_zero






