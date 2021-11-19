import numpy as np
import pandas as pd
import math
from numpy.random import seed
from numpy.random import rand
from numpy.random import choice
from scipy.stats import bernoulli 



class  SynWorld:
    def normalize(self,v):
        norm = np.linalg.norm(v)
        return v / norm
    def __init__(self,I,J,d,mu_inv,T,rho,n,repeat):
        self.type='syn'
        self.T=T
        self.mu_inv=mu_inv
        self.I=I  # number of job classes
        self.J=J  # number of server classes
        self.d=d  # dimension for context and model
        self.rho=rho
        self.n=n   #number of servers
        self.cl=dict()     #information of arrival jobs
        self.x=np.zeros((self.I,self.d)) # context of job classes
        self.y=np.zeros((self.J,self.d)) # context of server classses
        self.theta=np.zeros(self.d*self.d)  #true parameter for model
        self.sd=0.1  
        self.min_value=0  #min value of reward

#         np.random.seed(repeat+1)
        ##generate theta
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
            RV=bernoulli.rvs(round(self.rho.sum(),1)/self.mu_inv ,size=1)
            l= choice(range(self.I), 1, p=self.rho/self.rho.sum())[0] #class
            if RV==1:
                n= np.random.geometric(p=1/self.mu_inv, size=1)[0] #number of tasks
                self.cl[t]=[[l,n]]
            else:
                self.cl[t]=[[l,0]]
     
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
    
    
class RealWorld:
    def normalize(self,v):
        norm = np.linalg.norm(v)
        return v / norm
    def __init__(self,I,J,d,T,repeat):
        self.type='real'
        self.T=T
        self.I=I
        self.J=J
        self.d=d
        self.sd=0.1
        machine=pd.read_csv("./data/pre_machine.csv") #load machine (server)
        instance=pd.read_csv("./data/pre_instance.csv") #load instance
        cpi=pd.read_csv("./data/pre_cpi.csv")
        collection=pd.read_csv("./data/pre_collection.csv") #load collection (job)
        self.n=machine.groupby('cluster').size().reset_index(name='count')['count'].values     
        ##compute rho for each collection class
        lamb=collection.groupby('cluster').size().values/self.T
        mu_inv=collection['N'].median()
        self.rho=lamb*mu_inv
        
        self.cl=dict()
        self.x=np.zeros((self.I,self.d))
        self.y=np.zeros((self.J,self.d))
        self.theta=np.zeros(self.d*self.d)
        self.min_value=0
#         np.random.seed(repeat+1)

        ##reward distribution information
        self.avg_reward=np.zeros((self.I,self.J))
        self.var_reward=np.zeros((self.I,self.J))
        for i in range(self.I):
            for j in range(self.J):
                if ((i+1,j+1) in list(zip(cpi['collection_cluster'].values, cpi['machine_cluster'].values))):
                    temp=cpi.loc[(cpi['collection_cluster']==(i+1)) & (cpi['machine_cluster']==(j+1))]['1/cpi_mean']
                    self.avg_reward[i,j]=temp.values[0]
                    temp=cpi.loc[(cpi['collection_cluster']==i+1) & (cpi['machine_cluster']==j+1)]['1/cpi_variance']
                    self.var_reward[i,j]=temp.values[0]
                else:
                    self.avg_reward[i,j]=0
                    self.var_reward[i,j]=0

        ## context information
        df_temp=instance[['cpus','memory','cluster']].groupby(['cluster']).mean().reset_index()
        for i in range(self.I):
            self.x[i,0]=df_temp.loc[df_temp['cluster']==i+1]['cpus'].values[0]
            self.x[i,1]=df_temp.loc[df_temp['cluster']==i+1]['memory'].values[0]
            self.x[i,2]=1/self.x[i,0]
            self.x[i,3]=1/self.x[i,1]
        df_temp=machine[['cpus','memory','cluster']].drop_duplicates()
        for j in range(self.J):
            self.y[j,0]=df_temp.loc[df_temp['cluster']==j+1]['cpus'].values[0]
            self.y[j,1]=df_temp.loc[df_temp['cluster']==j+1]['memory'].values[0]
            self.y[j,2]=1/self.y[j,0]
            self.y[j,3]=1/self.y[j,1]
        
        ## arrival jobs
        t_prev=0
        for t in range(self.T):
            t_1=t*(10**6)*5
            df_temp=collection.loc[(collection['time']<t_1) & (collection['time']>=t_prev)]
            arriv_num=df_temp.shape[0]
            clus_list= df_temp['cluster'].values-1
            n_list=df_temp['N'].values   
            if arriv_num>0:
                for c,n in zip(clus_list,n_list):
                    if t not in self.cl.keys():
                        self.cl[t]=[[c,n]]
                    else:
                        self.cl[t].append([c,n])
            else:
                self.cl[t]=[[0,0]]
            t_prev=t_1   
    ## observe rewards
    def observe(self,i,j):
        reward=np.random.normal(self.avg_reward[int(i),j],np.sqrt(self.var_reward[int(i),j]))
        return reward
    ## mean rewards for regret bound
    def mean_reward(self,i,j):
        mean=self.avg_reward[int(i),j]
        return mean        
    