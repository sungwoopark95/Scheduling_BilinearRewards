from Environment import *
from Algorithm import *
from Oracle import *
from Preprocess import *
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
import sys


        
def save(exp_reward,exp_oracle_reward,queue_length,T,alg): ## save data
    Path("./result").mkdir(parents=True, exist_ok=True)
    
    cum_reward=np.cumsum(exp_reward.mean(axis=0))
    cum_reward_oracle=(np.array(range(T))+1)*exp_oracle_reward.mean()
    sd_regret=np.std(np.outer((np.array(range(T))+1),exp_oracle_reward).T-np.cumsum(exp_reward,axis=1),axis=0)
    np.savetxt('./result/reward_mean_'+alg+'.csv', cum_reward_oracle-cum_reward, delimiter=',')
    np.savetxt('./result/reward_sd_'+alg+'.csv',sd_regret, delimiter=',')
    queue_mean=queue_length.mean(axis=0)
    queue_sd=np.std(queue_length,axis=0)
    np.savetxt('./result/queue_mean_'+alg+'.csv', queue_mean, delimiter=',')
    np.savetxt('./result/queue_sd_'+alg+'.csv',queue_sd, delimiter=',')

def plot(repeat,com):  ## load and plot data
    if com==False:
        regret=np.loadtxt('./result/reward_mean_alg1.csv', delimiter=',')
        sd=np.loadtxt('./result/reward_sd_alg1.csv', delimiter=',')
        fig, ax = plt.subplots()
        ax.plot(range(T),regret, color='orange', label='Algorithm 1')
        ax.fill_between(range(T), (regret-1.96*sd/np.sqrt(repeat)), (regret+1.96*sd/np.sqrt(repeat)), color='orange', alpha=.1 )
        plt.xlabel('Time step t')
        plt.ylabel('Regret')
        plt.legend(loc='best')
        plt.savefig('./result/regret_plot.png')
        plt.clf()

        queue_mean=np.loadtxt('./result/queue_mean_alg1.csv', delimiter=',')
        queue_sd=np.loadtxt('./result/queue_sd_alg1.csv', delimiter=',')
        fig, ax = plt.subplots()
        ax.plot(range(T),queue_mean, color='orange', label='Algorithm 1')
        ax.fill_between(range(T), (queue_mean-1.96*queue_sd/np.sqrt(repeat)), (queue_mean+1.96*queue_sd/np.sqrt(repeat)), color='orange', alpha=.1 )
        plt.xlabel('Time step t')
        plt.ylabel('Queue Length')
        plt.legend(loc='best')
        plt.savefig('./result/queue_plot.png')
        plt.clf()
    
    elif com==True:
        regret1=np.loadtxt('./result/reward_mean_alg1.csv', delimiter=',')
        sd1=np.loadtxt('./result/reward_sd_alg1.csv', delimiter=',')
        regret2=np.loadtxt('./result/reward_mean_alg2.csv', delimiter=',')
        sd2=np.loadtxt('./result/reward_sd_alg2.csv', delimiter=',')
        fig, ax = plt.subplots()
        ax.plot(range(T),regret1, color='orange', label='Algorithm 1')
        ax.fill_between(range(T), (regret1-1.96*sd1/np.sqrt(repeat)), (regret1+1.96*sd1/np.sqrt(repeat)), color='orange', alpha=.1 )
        ax.plot(range(T),regret2,':', color='g', label='HXLB')
        ax.fill_between(range(T), (regret2-1.96*sd2/np.sqrt(repeat)), (regret2+1.96*sd2/np.sqrt(repeat)), color='g', alpha=.1 )
        plt.xlabel('Time step t')
        plt.ylabel('Regret')
        plt.legend(loc='best')
        plt.savefig('./result/regret_plot.png')
        plt.clf()

        queue_mean1=np.loadtxt('./result/queue_mean_alg1.csv', delimiter=',')
        queue_sd1=np.loadtxt('./result/queue_sd_alg1.csv', delimiter=',')
        queue_mean2=np.loadtxt('./result/queue_mean_alg2.csv', delimiter=',')
        queue_sd2=np.loadtxt('./result/queue_sd_alg2.csv', delimiter=',')
        fig, ax = plt.subplots()
        ax.plot(range(T),queue_mean1, color='orange', label='Algorithm 1')
        ax.fill_between(range(T), (queue_mean1-1.96*queue_sd1/np.sqrt(repeat)), (queue_mean1+1.96*queue_sd1/np.sqrt(repeat)), color='orange', alpha=.1 )
        ax.plot(range(T),queue_mean2,':', color='g', label='HXLB')
        ax.fill_between(range(T), (queue_mean2-1.96*queue_sd2/np.sqrt(repeat)), (queue_mean2+1.96*queue_sd2/np.sqrt(repeat)), color='g', alpha=.1 )
        plt.xlabel('Time step t')
        plt.ylabel('Queue Length')
        plt.legend(loc='best')
        plt.savefig('./result/queue_plot.png')
        plt.clf()

    
def run_syn(I,J,d,mu_inv,T,rho,n,gamma,V,repeat,util_arriv,load,env,com):    
    exp_reward1=np.zeros((repeat,T))
    exp_oracle_reward=np.zeros(repeat)
    queue_length1=np.zeros((repeat,T))
    exp_reward2=np.zeros((repeat,T))
    queue_length2=np.zeros((repeat,T))
    if load==False:
        for ind in range(repeat):
            print('repeat_ind:',ind+1)
            np.random.seed(ind+1)
            Env=SynWorld(I,J,d,mu_inv,T,rho,n,ind)  #generate environment
            algorithm1=Algorithm1(gamma,V,T,Env,util_arriv) #scheduling algorithm
            oracle=Oracle(Env) #oracle algorithm
            algorithm1.run()
            oracle.run()
            exp_oracle_reward[ind]=oracle.exp_oracle_reward
            exp_reward1[ind,:]=algorithm1.exp_reward 
            queue_length1[ind,:]=algorithm1.count_his
            if com==True:
                np.random.seed(ind+1)
                Env=SynWorld(I,J,d,mu_inv,T,rho,n,ind)  
                algorithm2=Algorithm2(gamma,V,T,Env,util_arriv) #algorithm for comparison
                algorithm2.run()
                exp_reward2[ind,:]=algorithm2.exp_reward 
                queue_length2[ind,:]=algorithm2.count_his

        save(exp_reward1,exp_oracle_reward,queue_length1,T,'alg1')
        if com==True:
            save(exp_reward2,exp_oracle_reward,queue_length2,T,'alg2')
    plot(repeat,com)
    
def run_real(I,J,d,T,gamma,repeat,load,env,ext,prep,com):
    
    exp_reward1=np.zeros((repeat,T))
    exp_oracle_reward=np.zeros(repeat)
    queue_length1=np.zeros((repeat,T))
    exp_reward2=np.zeros((repeat,T))
    queue_length2=np.zeros((repeat,T))
    Path("./data").mkdir(parents=True, exist_ok=True)
    if load==False:
        if ext==True:
            Preprocess.extraction()
        if prep==True:
            Preprocess.preprocess()
        V=setting_real(T,I,J)
        for ind in range(repeat):
            print('repeat_ind:',ind+1)
            np.random.seed(ind+1)
            Env=RealWorld(I,J,d,T,ind)   #generate environment
            algorithm1=Algorithm1(gamma,V,T,Env) #scheduling algorithm
            oracle=Oracle(Env) #oracle algorithm
            algorithm1.run()
            oracle.run()
            exp_oracle_reward[ind]=oracle.exp_oracle_reward
            if com==True:
                np.random.seed(ind+1)
                Env=RealWorld(I,J,d,T,ind)
                algorithm2=Algorithm2(gamma,V,T,Env) #algorithm for comparison
                algorithm2.run()
                exp_reward2[ind,:]=algorithm2.exp_reward 
                queue_length2[ind,:]=algorithm2.count_his
            exp_reward1[ind,:]=algorithm1.exp_reward 
            queue_length1[ind,:]=algorithm1.count_his
        save(exp_reward1,exp_oracle_reward,queue_length1,T,'alg1')
        if com==True:
            save(exp_reward2,exp_oracle_reward,queue_length2,T,'alg2')
    plot(repeat,com)
    
    
def setting_syn(I,J,n_tot,rho_tot,util_arriv,fix): ## set rho, n, V for syn experiment
    rho=np.zeros(I)
    n=np.zeros(J)
    if fix==False:
        for i in range(I):
            rho[i]=rho_tot/I
        for j in range(J):
            n[j]=max(int(n_tot/J),1)
    else: ##alternative setting with fixed job arrival rates and server capacities
        for i in range(I):
            rho[i]=0.1
        for j in range(J):
            n[j]=2
    rho_min=rho.min()
    rho_max=rho.max()
    rho_harm=len(rho)/np.sum(1.0/rho) 
    if util_arriv==True:
        V=math.sqrt((rho_min*rho_tot+rho_max**2/rho_min)*T/mu_inv)
    else:
        V=math.sqrt(I*T*(1+1/rho_harm)/mu_inv)
    return rho,n,V

def setting_real(T,I,J): ## set V for real experiment
    collection=pd.read_csv("./data/pre_collection.csv")
    lamb=collection.groupby('cluster').size().values/T
    mu_inv=collection['N'].median()
    rho=lamb*mu_inv
    rho_harm=len(rho)/np.sum(1.0/rho) 
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
        mu_inv=1 # mean job processing time
        rho_tot=1  #total arrival rate for processing time
        n_tot=8  #total departure rate for processing time
        gamma=1.2 # gamma>1
        repeat=10  #repeat number
        util_arriv=False #True: utilize traffic intensities for Algorithm 1
        load=False #True: load saved data without running the algorithm
        com=False #True: compare with other algorithm
        fix=False #True: alternative setting with fixed job arrival rates and server capacities 
        rho,n,V=setting_syn(I,J,n_tot,rho_tot,util_arriv,fix)
        run_syn(I,J,d,mu_inv,T,rho,n,gamma,V,repeat,util_arriv,load,env,com)
    
    elif opt=='real': #exp with real data
        env='RealWorld'
        I=5
        J=12
        d=4
        T=1100
        gamma=1.2
        repeat=10
        ext=False #True: extract real data
        prep=False #True: preprocess real data
        load=False
        com=False
        run_real(I,J,d,T,gamma,repeat,load,env,ext,prep,com)
    else:
        print('wrong input')
