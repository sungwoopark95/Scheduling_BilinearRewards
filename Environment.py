import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from numpy.random import choice
from scipy.stats import bernoulli 



class  Environment:
    def normalize(self,v):
        norm = np.linalg.norm(v)
        return v / norm
    def __init__(self,I,J,d,N,T,lamb,mu,repeat):
        self.T=T
        self.N=N
        self.I=I
        self.J=J
        self.d=d
        self.lamb=lamb
        self.mu=mu
        self.cl=np.zeros((self.T,2)) #information of arrival jobs
        self.x=np.zeros((self.I,self.d))
        self.y=np.zeros((self.J,self.d))
        self.theta=np.zeros(self.d*self.d)
        self.sd=0.1
        self.min_value=0

        np.random.seed(repeat+1)
        self.theta=np.random.uniform(low=self.min_value, high=1, size=self.d*self.d)
        self.theta=self.normalize(self.theta)
        ## job context generation
        for i in range(self.I):
            self.x[i]=np.random.uniform(self.min_value,1,self.d)
            self.x[i]=self.normalize(self.x[i])
        ## server context generation
        for j in range(self.J):
            self.y[j]=np.random.uniform(self.min_value,1,self.d)
            self.y[j]=self.normalize(self.y[j])

        ## arrival jobs
        for t in range(self.T):
            RV=bernoulli.rvs(round(self.lamb.sum())/self.N ,size=1)
            l= choice(range(self.I), 1, p=self.lamb/self.lamb.sum())[0] #class
            if RV==1:
                n= np.random.geometric(p=1/self.N, size=1)[0] #task number
                self.cl[t]=[l,n]
            else:
                self.cl[t]=[l,0]
    ## observe rewards
    def observe(self,i,j):
        z=np.outer(self.x[i], self.y[j]).flatten()
        reward=self.theta@z+np.random.normal(0, self.sd, 1)
        return reward
    ## mean rewards for regret bound
    def mean_reward(self,i,j):
        z=np.outer(self.x[i], self.y[j]).flatten()
        mean=self.theta@z
        return mean