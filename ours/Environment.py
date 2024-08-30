from typing import List
import numpy as np

class MedicalWorld:
    def __init__(self, I:int, J:int, T:int, lbdas:List[int], probs:List[List[float]], random_state:int):
        self.T = T # planning horizon
        self.I = I # number of jobs (types of patients)
        self.J = J # number of servers (hospitals)
        self.servers = [[] for _ in range(self.J)] # each nested list represents a queue for each server
        self.counts = np.zeros((self.I, self.J)) # number of patients of each type that are treated (I, J)
        self.ucbs = np.array([[np.iinfo(np.int32).max for _ in range(self.J)] for _ in range(self.I)]) # (I, J)
        self.t = 0 # each round
        self.min_reward = 0 # minimum reward
        self.lbdas = lbdas # mean of arrivals for each job; \lambda_i
        self.probs = probs # probability of assigning type i to type j, (I, J)
        self.gammas = np.zeros((self.I, self.J)) # Actual number of type i jobs assigned to server j at time t of policy \pi
        self.Qs = np.zeros((self.I, self.J)) # The number of type i jobs waiting in queue j
        self.Bs = np.zeros((self.I, self.J)) # The number of type i jobs served by server j in time t
        self.values = np.zeros((self.I, self.J)) # containing r_{ij}
    
    def simulate():
        ## Simulate arrival of patients
        return None
