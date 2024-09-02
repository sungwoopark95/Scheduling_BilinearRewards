import numpy as np
from typing import List

class BarrierFunction:
    def __init__(self, I:int, J:int, lbdas:np.ndarray, V:float, min_reward:float):
        self.I = I # number of patients
        self.J = J # number of hospitals
        # self.R = np.zeros((I, J)) # matrix containing r_{ij}(t)
        # self.R_bar = R_bar # matrix containing \bar{r}_{ij}(t-1)
        self.lbdas = lbdas # parameters of poisson distribution
        self.V = V # constant for optimization
        self.min_reward = min_reward
    
    def truncatedUCB(self, t:int, H:np.ndarray, R_bar:np.ndarray) -> np.ndarray:
        ## step1 function - returns r_{ij}(t)
        # H_{ij}(t) = \sum_{\tau=1}^t B_{ij}(\tau)
        # t : timestep
        # R_bar : matrix containing \bar{r}_{ij}(t-1)
        R = np.zeros((self.I, self.J)) # matrix containing r_{ij}(t)
        for i in range(self.I):
            for j in range(self.J):
                if H[i][j] == 0:
                    R[i][j] = 1
                else:
                    min_inside = R_bar[i][j] + np.sqrt((2 * np.log(t)) / H[i][j])
                    minimum = np.minimum(min_inside, 1)
                    R[i][j] = np.maximum(minimum, self.min_reward)
        return R

    def optimization(self) -> np.ndarray:
        ## step2 function - returns p_{ij}(t)
        pass

    # def update(self):
    #     ## update class variables such as R_bar
    #     pass
