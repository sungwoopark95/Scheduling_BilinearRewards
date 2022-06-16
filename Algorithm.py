'''
<Reference>
'minimize_solver_sol' function:
[1] https://github.com/waycan/QueueLearning/tree/master/QueueLearning
'''

import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from numpy.random import choice
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.stats import bernoulli 

class Algorithm1:
    def __init__(self,gamma,V,T,Env,w):
                 
        self.Env=Env
        self.x=self.Env.x        
        self.y=self.Env.y
        self.d=self.Env.d
        self.sd=self.Env.sd
        self.n=self.Env.n
        self.rho=self.Env.rho
        self.cl=self.Env.cl
        self.J=self.Env.J
        self.I=self.Env.I
        self.T=T
        self.V=V
        self.gamma=gamma
        self.min_value=0 #reward min value
        self.w=w
        if self.Env.type=='syn':
            self.zeta=1
            # self.zeta=np.sum(self.n)
        elif self.Env.type=='real':
            self.zeta=1
    
    def OFUL(self,i,j,A_inv,b,t):
        ##Compute reward estimators
        z=np.outer(self.x[i], self.y[j]).flatten()
        theta_hat=A_inv@b
        dim = self.d*self.d
        beta = self.sd*math.sqrt(dim*math.log((1+self.n.sum()*t)*self.T))+1
        p=z@theta_hat+np.sqrt(z@A_inv@z)*beta

        return p

    
     
    def minimize_solver_sol(self,ucb_m_gamma, class_ind, Q_len, n, V, epsilon):
        ## Compute assignment of jobs to servers

        if ucb_m_gamma.shape[0] == 0:
            return np.zeros(( ucb_m_gamma.shape[0],  ucb_m_gamma.shape[1]))

        num_job_class = ucb_m_gamma.shape[0]
        num_server = ucb_m_gamma.shape[1]
        p_job_server = np.ones((num_job_class, num_server))

        xinit = np.ones(num_job_class * num_server) 

        A = np.zeros((num_server, num_server * num_job_class))
        for j in range(num_server):
            A[j, j: (num_server * num_job_class): num_server] = 1

        func_val = []

        def obj_dynamic(x):
            f = 0.0
            for k,i in enumerate(class_ind):
                job_to_server_sum = np.sum(x[k*num_server: (k+1)*num_server])
                temp_sum = x[k*num_server: (k+1)*num_server].dot(ucb_m_gamma[k, :])
                f += ((self.w[i])/V) * Q_len[i]*np.log(job_to_server_sum + epsilon) + temp_sum  # add eps to avoid log(0)

            func_val.append(-f)

            return -f

        def ineq_const(x):
            return n - A @ x
        def ineq_const2(x):
            return x
        
        ineq_cons = [{'type': 'ineq','fun': ineq_const},
                     {'type': 'ineq','fun': ineq_const2}]

        bds = [(0, n[j]) for _ in range(num_job_class) for j in range(num_server)]

        res = minimize(obj_dynamic, x0=xinit, method='SLSQP',
                       constraints=ineq_cons,
                       bounds=bds)

        if res.success:
            p_opt = res.x
        else:
            raise(TypeError, "Cannot find a valid solution by SLSQP")

        for i in range(num_job_class):
            p_job_server[i, :] = p_opt[i*num_server: (i+1)*num_server]


        # return number of expected unit time to send
        return p_job_server 
    
    def run(self):
        ## running suggested Algorithm 1.
        
        A=self.zeta*np.identity(self.d*self.d)
        A_inv=np.linalg.inv(A)
        self.exp_reward=np.zeros(self.T)
        self.exp_reward_real=np.zeros(self.T)
        self.count_his=np.zeros(self.T)    ## for plot of queue length
        
        b=np.zeros(self.d*self.d)
        epsilon = np.power(10.0, -3.0)
        self.count_his=np.zeros(self.T)
        self.count_class_his=np.zeros((self.T,self.I))
        for t in range(self.T):
            self.inform=[]
            count_class=0  #number of remaining job classes at time t
            count=0 #number of remaining jobs at time t
            temp=0 #stored rewards for every assignment
            Q_len=np.zeros(self.I)
            if t%10==1:
                print('time:',t)
            for s in range(t+1):
                for ind in range(len(self.cl[s])):
                    if self.cl[s][ind][1]>0:
                        Q_len[int(self.cl[s][ind][0])]=Q_len[int(self.cl[s][ind][0])]+1
                        self.inform.append([s,self.cl[s][ind][0],ind]) #information of each job:[arrival time, class, index]
            class_ind=np.nonzero(Q_len)[0]
            count=int(Q_len.sum())
            if count!=0:
                self.ucb_est_jobs=np.zeros((len(class_ind), self.J))
                self.count_his[t]=count
                self.count_class_his[t]=Q_len
                for k,i in enumerate(class_ind):
                    for j in range(self.J):
                        self.ucb_est_jobs[k,j]=max(self.min_value,min(self.OFUL(i,j,A_inv,b,t),1))-self.gamma  
                while True:
                    try:
                        y= self.minimize_solver_sol(self.ucb_est_jobs, class_ind, Q_len, self.n, self.V, epsilon)
                        break
                    except:
                        print('exception')
                        epsilon=2*epsilon

                for k,i in enumerate(class_ind): ##Compute expected rewards for regret bound
                    for j in range(self.J): 
                        temp=temp+y[k,j]*self.Env.mean_reward(i,j)
                self.exp_reward[t]=temp    
                N=np.zeros((count,self.J))
                for i in range(count):
                    k=np.where(class_ind==int(self.inform[i][1]))
                    N[i,:]=y[k,:]/Q_len[int(self.inform[i][1])] ##expected assignemnt for each job

            ## assign jobs and get feedback
            z_his=[]
            reward_his=[]
            reward_his_real=[]
            if count!=0:
                for j in range(self.J):  
                    remain=max(0,1-N[:,j].sum()/self.n[j])
                    prob=np.append(N[:,j]/self.n[j],remain)
                    prob=prob/prob.sum()
                    for v in range(int(self.n[j])):
                        i=choice(range(count+1), 1, p=prob)[0]
                        if i==count:
                            continue
                        arriv_time=self.inform[i][0]
                        ind=self.inform[i][2]
                        remain_num=self.cl[arriv_time][ind][1]                        
                        class_=self.inform[i][1]
                        z=np.outer(self.x[int(class_)], self.y[j]).flatten()
                        reward=self.Env.observe(int(class_),j) #get rewards
                        z_his.append(z)
                        reward_his.append(reward)
                        self.cl[arriv_time][ind][1]=self.cl[arriv_time][ind][1]-1
            
            ## update parameters       
            G=A_inv
            if len(z_his)>0:
                z_his=np.array(z_his)
                reward_his=np.array(reward_his)
                for l in range(z_his.shape[0]):
                    A=A+np.outer(z_his[l],z_his[l])
                    b=b+z_his[l]*reward_his[l]
                G=G-(G@np.outer(z_his[l],z_his[l])@G)/(1+z_his[l]@G@z_his[l]) 
            A_inv=G
            
            
            
class Algorithm2: #for comparison
    def __init__(self,gamma,V,T,Env):
                 
        self.Env=Env
        self.x=self.Env.x        
        self.y=self.Env.y
        self.d=self.Env.d
        self.sd=self.Env.sd
        self.n=self.Env.n
        self.rho=self.Env.rho
        self.cl=self.Env.cl
        self.J=self.Env.J
        self.I=self.Env.I
        self.T=T
        self.V=V
        self.gamma=gamma
        self.min_value=0 #reward min value
        

    
     
    def minimize_solver_sol(self,ucb_m_gamma, n, V, epsilon):
        ## Compute assignment of jobs to servers

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
            for i in range(num_job):
                prob_to_server_sum = np.sum(x[i*num_server: (i+1)*num_server])
                temp_sum = x[i*num_server: (i+1)*num_server].dot(ucb_m_gamma[i, :])
                f += (1/V) * np.log(prob_to_server_sum + epsilon) + temp_sum  # add eps to avoid log(0)

            func_val.append(-f)

            return -f

        def ineq_const(x):
            return n - A @ x
        def ineq_const2(x):
            return x
        
        ineq_cons = [{'type': 'ineq','fun': ineq_const},
                     {'type': 'ineq','fun': ineq_const2}]

        bds = [(0, n[j]) for _ in range(num_job) for j in range(num_server)]

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
        
        self.exp_reward=np.zeros(self.T)
        self.count_his=np.zeros(self.T)
        self.count_class_his=np.zeros((self.T,self.I))

        ind_max=0
        for s in range(self.T):
            if ind_max < len(self.cl[s]):
                ind_max=len(self.cl[s])
        
        self.his=np.zeros((self.T,ind_max,self.J,2))## est reward mean, selected num
        epsilon = np.power(10.0, -3.0)

                    
        for t in range(self.T):
            self.inform=[]
            count=0  #number of remaining jobs at time t
            temp=0 #stored rewards for every assignment
            Q_len=np.zeros(self.I)
            if t%10==1:
                print('time:',t)
            for s in range(t+1):
                for ind in range(len(self.cl[s])):
                    if self.cl[s][ind][1]>0:
                        Q_len[int(self.cl[s][ind][0])]=Q_len[int(self.cl[s][ind][0])]+1
                        
                        self.inform.append([s,self.cl[s][ind][0],ind]) #information of each job:[arrival time, class, index]
            count=int(Q_len.sum())
            self.ucb_est_jobs=np.zeros((count, self.J))
            self.count_his[t]=count

            for i in range(count):
                for j in range(self.J):
                    arriv_time=self.inform[i][0]
                    ind=self.inform[i][2]
                    if self.his[arriv_time,ind,j,1]==0:
                        ucb=1
                    else:
                        ucb=self.his[arriv_time,ind,j,0]+math.sqrt(2*math.log(self.his[arriv_time,ind,j,1])/self.his[arriv_time,ind,j,1])
                    self.ucb_est_jobs[i,j]=max(self.min_value,min(ucb,1))-self.gamma

            
            while True:
                try:
                    N= self.minimize_solver_sol(self.ucb_est_jobs, self.n, self.V, epsilon)
                    break
                except:
                    print('exception')
                    epsilon=2*epsilon
            
            for i in range(count): ##Compute expected rewards for regret bound
                for j in range(self.J): 
                    z=np.outer(self.x[int(self.inform[i][1])],self.y[j]).flatten()
                    temp=temp+N[i,j]*self.Env.mean_reward(int(self.inform[i][1]),j)
            self.exp_reward[t]=temp    
            
            ## assign jobs and get feedback
            if count!=0:
                for j in range(self.J):  
                    remain=max(0,1-N[:,j].sum()/self.n[j])
                    prob=np.append(N[:,j]/self.n[j],remain)
                    prob=prob/prob.sum()
                    for v in range(int(self.n[j])):
                        i=choice(range(count+1), 1, p=prob)[0]
                        if i==count:
                            continue
                        arriv_time=self.inform[i][0]
                        ind=self.inform[i][2]
                        remain_num=self.cl[arriv_time][ind][1]                        
                        class_=self.inform[i][1]
                        
                        temp=self.his[arriv_time,ind,j,1].copy()
                        self.his[arriv_time,ind,j,1]=self.his[arriv_time,ind,j,1]+1
                        reward=self.Env.observe(int(class_),j) #get rewards
                        self.his[arriv_time,ind,j,0]=(self.his[arriv_time,ind,j,0]*temp+reward)/self.his[arriv_time,ind,j,1]   
                        self.cl[arriv_time][ind][1]=self.cl[arriv_time][ind][1]-1
