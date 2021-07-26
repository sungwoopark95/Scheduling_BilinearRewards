'''
<Reference>
'oracle_solver' function:
[1] https://github.com/waycan/QueueLearning/tree/master/QueueLearning
'''
import numpy as np
import math
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

class Oracle:
    def __init__(self,Env):    
        self.Env=Env
        self.I=self.Env.I
        self.J=self.Env.J
        self.x=self.Env.x
        self.y=self.Env.y
        self.lamb=self.Env.lamb
        self.mu=self.Env.mu

    def oracle_solver(self,reward_mean,lamb, mu):
        ## oracle policy for regret

        if reward_mean.shape[0] == 0:
            return np.zeros(( reward_mean.shape[0],  reward_mean.shape[1]))

        num_job = reward_mean.shape[0]
        num_server = reward_mean.shape[1]
        p_job_server = np.ones((num_job, num_server))

        xinit = np.ones(num_job * num_server)   

        A = np.zeros((num_server, num_server * num_job))
        for j in range(num_server):
            A[j, j: (num_server * num_job): num_server] = 1

        func_val = []

        def obj_dynamic(x):
            f = 0.0
            for i in range(num_job):
                temp_sum = lamb[i]*x[i*num_server: (i+1)*num_server].dot(reward_mean[i, :])

                f += temp_sum  

            func_val.append(-f)

            return -f
        b=np.ones(num_job * num_server)
        for i in range(num_job):
            b[i*num_server:(i+1)*num_server]=self.lamb[i]
            
        def eq_const(x):
            c=np.ones(num_job)
            
            return c-x.reshape(num_job,num_server)@np.ones(num_server)
        
        def ineq_const(x):
            return self.mu - A @ (x*b)

        ineq_cons = [{'type': 'ineq',
                     'fun': ineq_const}, {'type': 'eq',
                     'fun': eq_const}]

        bds = [(0, 1) for _ in range(num_job) for _ in range(num_server)]

        res = minimize(obj_dynamic, x0=xinit, method='SLSQP',
                       constraints=ineq_cons,
                       bounds=bds)

        if res.success:
            p_opt = res.x
        else:
            raise(TypeError, "Cannot find a valid solution by SLSQP")

        for i in range(num_job):
            p_job_server[i, :] = p_opt[i*num_server: (i+1)*num_server]

        # return probability to send each job 
        return p_job_server 
    
    def run(self):  
        self.exp_oracle_reward=0   
        prob=np.ones((self.I,self.J))/self.J
        C=np.zeros((self.I,self.J))
        for i in range(self.I):
            for j in range(self.J):
                C[i,j]=self.Env.mean_reward(i,j)
        prob=self.oracle_solver(C,self.lamb, self.mu).reshape(self.I,self.J)
        for i in range(self.I):
            for j in range(self.J):
                  self.exp_oracle_reward=self.exp_oracle_reward+self.lamb[i]*prob[i,j]*self.Env.mean_reward(i,j)
    
        