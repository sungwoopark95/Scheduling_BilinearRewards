import numpy as np
from typing import List

class BarrierFunction:
    def __init__(self, I:int, J:int, H:np.ndarray, R_bar:np.ndarray, lbdas:np.ndarray, V:float):
        self.I = I # number of patients
        self.J = J # number of hospitals
        self.H = H # matrix of the number of patients treated
        self.R = np.zeros((I, J)) # matrix containing r_{ij}(t)
        self.R_bar = R_bar # matrix containing \bar{r}_{ij}(t-1)
        self.lbdas = lbdas # parameters of poisson distribution
        self.V = V # constant for optimization
    
    def truncatedUCB(self) -> np.ndarray:
        # step1 function - returns r_{ij}(t)
        pass

    def optimization(self) -> np.ndarray:
        # step2 function - returns p_{ij}(t)
        pass
