'''
<Reference>
'minimize_solver_sol' function:
[1] https://github.com/waycan/QueueLearning/tree/master/QueueLearning
'OFUL' function:
[2] https://www.cvxpy.org/examples/basic/socp.html
'''

import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from numpy.random import choice
import cvxpy as cp
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.stats import bernoulli 

class Algorithm:
    def __init__(self,gamma,V,T,util_arriv,Env):
                 
        self.Env=Env
        self.x=self.Env.x        
        self.y=self.Env.y
        self.d=self.Env.d
        self.sd=self.Env.sd
        self.mu=self.Env.mu
        self.lamb=self.Env.lamb
        self.cl=self.Env.cl
        self.J=self.Env.J
        self.I=self.Env.I
        self.T=T
        self.V=V
        self.gamma=gamma
        self.bool_=util_arriv #True for utilizing arriv rates
        self.min_value=0 ##reward min value
        
    def normalize(self,v):
        norm = np.linalg.norm(v)
        return v / norm
    
    def OFUL(self,i,j,A,b,t):
        ###Compute reward estimators
        
        z=np.outer(self.x[int(i)], self.y[j]).flatten()
        theta_hat=self.A_inv@b
        dim = self.d*self.d
        beta = self.sd*math.sqrt(dim*math.log((1+self.mu.sum()*t)*self.T))+1
        x0 = self.normalize(np.random.randn(dim)) #initial theta


        # Define and solve the CVXPY problem.
        x = cp.Variable(dim) #theta variable
        # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
        soc_constraints = [
              cp.SOC( beta, self.A_chol.T @ x )
        ]
        prob = cp.Problem(cp.Maximize(z.T@x),
                          soc_constraints)
        prob.solve()
        p=prob.value+z.T@theta_hat  #UCB value
        return p

    
     
    def minimize_solver_sol(self,ucb_m_gamma, mu, V):
        ### Compute job to server assignment

        if ucb_m_gamma.shape[0] == 0:
            return np.zeros(( ucb_m_gamma.shape[0],  ucb_m_gamma.shape[1]))

        num_job = ucb_m_gamma.shape[0]
        num_server = ucb_m_gamma.shape[1]
        p_job_server = np.ones((num_job, num_server))

        xinit = np.ones(num_job * num_server) 

        A = np.zeros((num_server, num_server * num_job))
        for j in range(num_server):
            A[j, j: (num_server * num_job): num_server] = 1

        func_val = []

        def obj_dynamic(x):
            f = 0.0
            epsilon = np.power(10.0, -3.0)
            for i in range(num_job):
                prob_to_server_sum = np.sum(x[i*num_server: (i+1)*num_server])
                temp_sum = x[i*num_server: (i+1)*num_server].dot(ucb_m_gamma[i, :])
                if self.bool_==True:
                    f += (self.lamb[int(self.inform[i][1])]/V) * np.log(prob_to_server_sum + epsilon) + temp_sum  # add eps to avoid log(0)
                else:
                    f += (1/V) * np.log(prob_to_server_sum + epsilon) + temp_sum  # add eps to avoid log(0)

            func_val.append(-f)

            return -f

        def ineq_const(x):
            return mu - A @ x
        def ineq_const2(x):
            return x
        
        ineq_cons = [{'type': 'ineq','fun': ineq_const},
                     {'type': 'ineq','fun': ineq_const2}]

        bds = [(0, mu[j]) for _ in range(num_job) for j in range(num_server)]

        res = minimize(obj_dynamic, x0=xinit, method='SLSQP',
                       constraints=ineq_cons,
                       bounds=bds)

        if res.success:
            p_opt = res.x
        else:
            raise(TypeError, "Cannot find a valid solution by SLSQP")

        for i in range(num_job):
            p_job_server[i, :] = p_opt[i*num_server: (i+1)*num_server]


        # return number of expected unit time to send
        return p_job_server 
    
    def run(self):
        ## running suggested Algorithm 1.
        
        A=np.identity(self.d*self.d)
        self.A_inv=np.linalg.inv(A)
        self.A_chol=np.linalg.cholesky(A)        
        self.exp_reward=np.zeros(self.T)
        self.count_his=np.zeros(self.T)

        b=np.zeros(self.d*self.d)
        p=np.zeros((self.d,self.d))

        for t in range(self.T):
            self.inform=[]
            count=0  #number of remaining jobs at time t
            temp=0 #stored rewards for every assignment
            if t%10==1:
                print('time:',t)
            for s in range(t+1):
                if self.cl[s,1]>0:
                    count+=1
                    self.inform.append([s,self.cl[s,0]]) #information of each job:[arrival time, class]
            self.ucb_est_jobs=np.zeros((count, self.J))
            self.count_his[t]=count

            for i in range(count):
                for j in range(self.J):  
                    self.ucb_est_jobs[i,j]=max(self.min_value,min(self.OFUL(self.inform[i][1],j,A,b,t),1))-self.gamma  
            n=self.minimize_solver_sol(self.ucb_est_jobs, self.mu, self.V)    
            for i in range(count): ##Compute expected rewards for regret bound
                for j in range(self.J): 
                    z=np.outer(self.x[int(self.inform[i][1])],self.y[j]).flatten()
                    temp=temp+n[i,j]*self.Env.mean_reward(int(self.inform[i][1]),j)
            self.exp_reward[t]=temp    
            ## assign jobs and get feedback
            z_his=[]
            reward_his=[]
            if count!=0:
                for j in range(self.J):  
                    remain=max(0,1-n[:,j].sum()/self.mu[j])
                    prob=np.append(n[:,j]/self.mu[j],remain)
                    prob=prob/prob.sum()
                    for v in range(int(self.mu[j])):
                        i=choice(range(count+1), 1, p=prob)[0]
                        if i==count:
                            continue
                        arriv_time=self.inform[i][0]
                        remain_num=self.cl[arriv_time,1]
                        class_=self.inform[i][1]
                        if remain_num>0:
                            z=np.outer(self.x[int(class_)], self.y[j]).flatten()
                            reward=self.Env.observe(int(class_),j) #get rewards
                            z_his.append(z)
                            reward_his.append(reward)
                            self.cl[arriv_time,1]=self.cl[arriv_time,1]-1
            ## update parameters        
            if len(z_his)>0:
                z_his=np.array(z_his)
                reward_his=np.array(reward_his)
                for l in range(z_his.shape[0]):
                    A=A+np.outer(z_his[l],z_his[l])
                    b=b+z_his[l]*reward_his[l]
                self.A_inv=np.linalg.inv(A)
                self.A_chol=np.linalg.cholesky(A)
    