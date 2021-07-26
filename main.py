from Environment import *
from Algorithm import *
from Oracle import *
import numpy as np
import math
import matplotlib.pyplot as plt


        
def save(exp_reward,exp_oracle_reward,queue_length,T): ## save data
    cum_reward=np.cumsum(exp_reward.mean(axis=0))
    cum_reward_oracle=(np.array(range(T))+1)*exp_oracle_reward.mean()
    sd_regret=np.std(np.outer((np.array(range(T))+1),exp_oracle_reward).T-exp_reward,axis=0)
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
    plt.xlabel('T')
    plt.ylabel('Regret')
    plt.legend(loc='best')
    plt.savefig('./result/regret_plot.png')
    plt.show()
    
    queue_mean=np.loadtxt('./result/queue_mean.csv', delimiter=',')
    queue_sd=np.loadtxt('./result/queue_sd.csv', delimiter=',')
    fig, ax = plt.subplots()
    ax.plot(range(T),queue_mean, color='orange', label='Algorithm 1')
    ax.fill_between(range(T), (queue_mean-1.96*queue_sd/np.sqrt(repeat)), (queue_mean+1.96*queue_sd/np.sqrt(repeat)), color='orange', alpha=.1 )
    plt.xlabel('T')
    plt.ylabel('Queue Length')
    plt.legend(loc='best')
    plt.savefig('./result/queue_plot.png')
    plt.show()

    
def run(I,J,d,N,T,lamb,mu,gamma,V,repeat,util_arriv,load):
    exp_reward=np.zeros((repeat,T))
    exp_oracle_reward=np.zeros(repeat)
    queue_length=np.zeros((repeat,T))
    if load==False:
        for ind in range(repeat):
            print('repeat_ind:',ind+1)
            Env=Environment(I,J,d,N,T,lamb,mu,ind)  #generated environment
            algorithm=Algorithm(gamma,V,T,util_arriv,Env) #suggested algorithm
            oracle=Oracle(Env) #oracle algorithm
            algorithm.run()
            oracle.run()
            exp_reward[ind,:]=algorithm.exp_reward 
            exp_oracle_reward[ind]=oracle.exp_oracle_reward
            queue_length[ind,:]=algorithm.count_his
        save(exp_reward,exp_oracle_reward,queue_length,T)
    plot(repeat)
    
def setting(I,J,mu_tot,lamb_tot,bool_): ## setting for lamb, mu, V
    lamb=np.zeros(I)
    mu=np.zeros(J)
    for i in range(I):
        lamb[i]=lamb_tot/I
    for j in range(J):
        mu[j]=max(int(mu_tot/J),1)
    lamb_min=lamb.min()
    if bool_==True:
        V=math.sqrt(lamb_min*T/N)
    else:
        V=math.sqrt(I*T/(lamb_min*N))
    return lamb,mu,V

if __name__ == "__main__":   
    
    I=10 # number of job classes
    J=4  # number of servers
    T=100 # time horizon
    d=2 # dimension for context vectors
    N=int(T**(1/3)) # mean job processing time
    lamb_tot=1  #total arrival rate for processing time
    mu_tot=8  #total departure rate for processing time
    gamma=1.2 # gamma>1
    util_arriv=False #True: utilize arrival rates for Algorithm 1
    load=False #True: load saved data without running the algorithm
    repeat=10  #repeat number
    
    lamb,mu,V=setting(I,J,mu_tot,lamb_tot,util_arriv)
    run(I,J,d,N,T,lamb,mu,gamma,V,repeat,util_arriv,load)
    