import numpy as np
from typing import List
from Algorithm import *

def sample_lambdas(high:int, size:int, epsilon:float=0.1) -> np.ndarray[float]:
    """
    high: upper bound of samples; set as the total number of patients
    size: the number of parameters to be sampled; set as the total number of servers
    epsilon: adjustment constant to make the sum of the samples less than the size
    """
    lambdas = np.random.randint(low=1, high=high, size=size)
    lambdas = lambdas / (np.sum(lambdas) + epsilon) # to make np.sum(lambdas) < size
    lambdas = lambdas * size 
    return lambdas

if __name__ == "__main__":
    ## set the seed
    SEED = 777
    np.random.seed(SEED)

    ## Initialize necessary values
    I = 100 # total number of jobs (patients)
    J = 10 # total number of servers (hospitals)
    lbdas = sample_lambdas(I, J)
    T = 2000 # total horizon
    true_mean_reward = np.random.uniform(low=0, high=1, size=J)
    min_reward = np.amin(true_mean_reward)

    ## Initialize matrices
    Gamma = np.zeros((I, J)) # Gamma[i][j]: the number of patients of type i arrived at the hospital j
    Q = np.zeros((I, J)) # Q[i][j]: the number of waiting patients of type i in the hospital j
    B = np.zeros((I, J)) # B[i][j]: the number of treated patients of type i in the hospital j
    H = np.zeros((I, J)) # H_{ij}(t) = \sum_{\tau=1}^t B_{ij}(\tau)
    # R = np.zeros((I, J)) # matrix for the rewards
    R_bar = np.zeros((I, J)) # matrix for the average rewards
    
    ## run simulator
    # 1. for each time
    # 2. initialize the algorithm class
    # 3. implement the step 1
    # 4. implement the step 2
    # 5. with returned r_{ij}(t) and p_{ij}(t) implement update - the step3 in the paper
