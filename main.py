from Environment import *
from Algorithm import *
from Oracle import *
from Preprocess import *
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
import sys


        
def save(exp_reward,exp_oracle_reward,queue_length,T): ## save data
    Path("./result").mkdir(parents=True, exist_ok=True)

    
    cum_reward=np.cumsum(exp_reward.mean(axis=0))
    cum_reward_oracle=(np.array(range(T))+1)*exp_oracle_reward.mean()
    sd_regret=np.std(np.outer((np.array(range(T))+1),exp_oracle_reward).T-np.cumsum(exp_reward,axis=1),axis=0)
    np.savetxt('./result/reward_mean.csv', cum_reward_oracle-cum_reward, delimiter=',')
    np.savetxt('./result/reward_sd.csv',sd_regret, delimiter=',')
    
    queue_mean=queue_length.mean(axis=0)
    queue_sd=np.std(queue_length,axis=0)
    np.savetxt('./result/queue_mean.csv', queue_mean, delimiter=',')
    np.savetxt('./result/queue_sd.csv',queue_sd, delimiter=',')

def plot(repeat):  ## load and plot data

    regret=np.loadtxt('./result/reward_mean.csv', delimiter=',')
    sd=np.loadtxt('./result/reward_sd.csv', delimiter=',')
    fig, ax = plt.subplots()
    ax.plot(range(T),regret, color='orange', label='Algorithm 1')
    ax.fill_between(range(T), (regret-1.96*sd/np.sqrt(repeat)), (regret+1.96*sd/np.sqrt(repeat)), color='orange', alpha=.1 )
    plt.xlabel('Time step t')
    plt.ylabel('Regret')
    plt.legend(loc='best')
    plt.savefig('./result/regret_plot.png')
    plt.show()
    plt.clf()
    
    queue_mean=np.loadtxt('./result/queue_mean.csv', delimiter=',')
    queue_sd=np.loadtxt('./result/queue_sd.csv', delimiter=',')
    fig, ax = plt.subplots()
    ax.plot(range(T),queue_mean, color='orange', label='Algorithm 1')
    ax.fill_between(range(T), (queue_mean-1.96*queue_sd/np.sqrt(repeat)), (queue_mean+1.96*queue_sd/np.sqrt(repeat)), color='orange', alpha=.1 )
    plt.xlabel('Time step t')
    plt.ylabel('Queue Length')
    plt.legend(loc='best')
    plt.savefig('./result/queue_plot.png')
    plt.show()
    plt.clf()

    
def run_syn(I,J,d,mu_inv,T,rho,n,gamma,V,repeat,util_arriv,load,env):    
    exp_reward=np.zeros((repeat,T))
    exp_oracle_reward=np.zeros(repeat)
    queue_length=np.zeros((repeat,T))
    if load==False:
        for ind in range(repeat):
            print('repeat_ind:',ind+1)
            Env=SynWorld(I,J,d,mu_inv,T,rho,n,ind)  #generate environment
            algorithm=Algorithm(gamma,V,T,Env,util_arriv) #scheduling algorithm
            oracle=Oracle(Env) #oracle algorithm
            algorithm.run()
            oracle.run()
            exp_reward[ind,:]=algorithm.exp_reward 
            exp_oracle_reward[ind]=oracle.exp_oracle_reward
            queue_length[ind,:]=algorithm.count_his
        save(exp_reward,exp_oracle_reward,queue_length,T)
    plot(repeat)
    
def run_real(I,J,d,T,gamma,repeat,load,env,ext=False,prep=False):
    
    exp_reward=np.zeros((repeat,T))
    exp_oracle_reward=np.zeros(repeat)
    queue_length=np.zeros((repeat,T))
    Path("./data").mkdir(parents=True, exist_ok=True)
    if load==False:
        if ext==True:
            Preprocess.extraction()
        if prep==True:
            Preprocess.preprocess()
        V=setting_real(T,I,J)
        for ind in range(repeat):
            print('repeat_ind:',ind+1)
            Env=RealWorld(I,J,d,T,ind)   #generate environment
            algorithm=Algorithm(gamma,V,T,Env)  #scheduling algorithm
            oracle=Oracle(Env) #oracle algorithm
            algorithm.run()
            oracle.run()
            exp_reward[ind,:]=algorithm.exp_reward 
            exp_oracle_reward[ind]=oracle.exp_oracle_reward
            queue_length[ind,:]=algorithm.count_his
        save(exp_reward,exp_oracle_reward,queue_length,T)
    plot(repeat)
    
    
def setting_syn(I,J,n_tot,rho_tot,util_arriv): ## set rho, n, V for syn exp
    rho=np.zeros(I)
    n=np.zeros(J)
    for i in range(I):
        rho[i]=rho_tot/I
    for j in range(J):
        n[j]=max(int(n_tot/J),1)
    rho_min=rho.min()
    rho_max=rho.max()
    rho_harm=len(rho)/np.sum(1/rho) 
    if util_arriv==True:
        V=math.sqrt((rho_min*rho_tot+rho_max**2/rho_min)*T/mu_inv)
    else:
        V=math.sqrt(I*T*(1+1/rho_harm)/mu_inv)
    return rho,n,V
def setting_real(T,I,J): ## set V for real exp
    
    collection=pd.read_csv("./data/pre_collection.csv")
    lamb=collection.groupby('cluster').size().values/T
    mu_inv=collection['N'].median()
    rho=lamb*mu_inv
    rho_harm=len(rho)/np.sum(1/rho) 
    V=math.sqrt(I*T*(1+1/rho_harm)/mu_inv)
    
    return V

    
if __name__ == "__main__":   
    opt = str(sys.argv[1]) #input; 'syn' or 'real'
    
    if opt=='syn':  #exp with synthetic data
        env='SynWorld'
        I=10 # number of job classes
        J=2  # number of servers
        T=100 # time horizon
        d=2 # dimension for context vectors
        mu_inv=T**(1/3) # mean job processing time
        rho_tot=1  #total arrival rate for processing time
        n_tot=8  #total departure rate for processing time
        gamma=1.2 # gamma>1
        util_arriv=False #True: utilize traffic intensities for Algorithm 1
        load=False #True: load saved data without running the algorithm
        repeat=10  #repeat number
        rho,n,V=setting_syn(I,J,n_tot,rho_tot,util_arriv)
        run_syn(I,J,d,mu_inv,T,rho,n,gamma,V,repeat,util_arriv,load,env)
    
    elif opt=='real': #exp with real data
        env='RealWorld'
        I=5
        J=12
        d=4
        T=1100
        gamma=1.2
        ext=False #True: extract real data
        prep=False #True: preprocess real data
        repeat=5
        load=False
        run_real(I,J,d,T,gamma,repeat,load,env,ext,prep)
    else:
        print('wrong input')
