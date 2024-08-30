from Environment import *
from Algorithm import *
from Oracle import *
from Preprocess import *
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
import sys


        
def save(exp_reward,exp_oracle_reward,queue_length,queue_class_length,T,name,c,bool_=False): ## save data
    Path("./result").mkdir(parents=True, exist_ok=True)
    cum_reward=np.cumsum(exp_reward.mean(axis=0))
    cum_reward_oracle=(np.array(range(T))+1)*exp_oracle_reward.mean()
    if bool_==False:
        sd_regret=np.std(np.outer((np.array(range(T))+1),exp_oracle_reward).T-np.cumsum(exp_reward,axis=1),axis=0)
    else:
        sd_regret=np.std(np.cumsum(exp_reward,axis=1),axis=0)
    if bool_==False:
        np.savetxt('./result/reward_mean_'+name+'.csv', cum_reward_oracle-cum_reward, delimiter=',')
    else:
        np.savetxt('./result/reward_mean_'+name+'.csv',cum_reward, delimiter=',')
    np.savetxt('./result/reward_sd_'+name+'.csv',sd_regret, delimiter=',')
    queue_mean=queue_length.mean(axis=0)
    queue_sd=np.std(queue_length,axis=0)
    queue_class_mean=queue_class_length.mean(axis=0)@c
    queue_class_sd=np.std(queue_class_length@c,axis=0)
    queue_i_mean=queue_class_length.mean(axis=0)
    queue_i_sd=np.std(queue_class_length,axis=0)
    np.savetxt('./result/queue_i_mean_'+name+'.csv', queue_i_mean, delimiter=',')
    np.savetxt('./result/queue_i_sd_'+name+'.csv', queue_i_sd, delimiter=',')
    np.savetxt('./result/queue_mean_'+name+'.csv', queue_mean, delimiter=',')
    np.savetxt('./result/queue_class_mean_'+name+'.csv', queue_class_mean, delimiter=',')
    np.savetxt('./result/queue_sd_'+name+'.csv',queue_sd, delimiter=',')
    np.savetxt('./result/queue_class_sd_'+name+'.csv',queue_class_sd, delimiter=',')

    
    
    
def plot_com(repeat,T,I,J,mu_inv,rho,n,bool_):
    alg_list=['alg1','com']
    load=rho/n
    regret=np.zeros((len(alg_list),T))
    sd=np.zeros((len(alg_list),T))
    name='alg1'+'_'+'T'+str(T)+'I'+str(I)+'J'+str(J)+'mu_inv'+str(mu_inv)+'rho'+str(rho)+'n'+str(n)+'bool'+str(bool_)
    regret[0]=np.loadtxt('./result/reward_mean_'+name+'.csv', delimiter=',')
    sd[0]=np.loadtxt('./result/reward_sd_'+name+'.csv', delimiter=',')
    name='com'+'_'+'T'+str(T)+'I'+str(I)+'J'+str(J)+'mu_inv'+str(mu_inv)+'rho'+str(rho)+'n'+str(n)+'bool'+str(bool_)
    regret[1]=np.loadtxt('./result/reward_mean_'+name+'.csv', delimiter=',')
    sd[1]=np.loadtxt('./result/reward_sd_'+name+'.csv', delimiter=',')    
    
    fig, ax = plt.subplots()
    ax.tick_params(labelsize=15)
    plt.rc('legend',fontsize=18)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.get_offset_text().set_fontsize(15)
    plt.gcf().subplots_adjust(bottom=0.15)   
    ax.plot(range(T),regret[0], color='lightsalmon', label='SABR')
    ax.plot(range(T),regret[1],'-.', color='royalblue', label='UGDA-OL')
    ax.fill_between(range(T), (regret[0]-1.96*sd[0]/np.sqrt(repeat)), (regret[0]+1.96*sd[0]/np.sqrt(repeat)), color='lightsalmon', alpha=.1 )
    ax.fill_between(range(T), (regret[1]-1.96*sd[1]/np.sqrt(repeat)), (regret[1]+1.96*sd[1]/np.sqrt(repeat)), color='royalblue', alpha=.1 )

    plt.xlabel('Time step t',fontsize=18)
    plt.ylabel('Regret',fontsize=18)
    plt.legend(loc='upper left')
    plt.savefig('./result/'+'regret_com_'+str(load)+'_'+str(bool_)+'_plot.png',bbox_inches = "tight",dpi=300)
    plt.show()
    plt.clf()


    queue_mean=np.zeros((len(alg_list),T))
    queue_sd=np.zeros((len(alg_list),T))
    name='alg1'+'_'+'T'+str(T)+'I'+str(I)+'J'+str(J)+'mu_inv'+str(mu_inv)+'rho'+str(rho)+'n'+str(n)+'bool'+str(bool_)
    queue_mean[0]=np.loadtxt('./result/queue_mean_'+name+'.csv', delimiter=',')
    queue_sd[0]=np.loadtxt('./result/queue_sd_'+name+'.csv', delimiter=',')
    name='com'+'_'+'T'+str(T)+'I'+str(I)+'J'+str(J)+'mu_inv'+str(mu_inv)+'rho'+str(rho)+'n'+str(n)+'bool'+str(bool_)
    queue_mean[1]=np.loadtxt('./result/queue_mean_'+name+'.csv', delimiter=',')
    queue_sd[1]=np.loadtxt('./result/queue_sd_'+name+'.csv', delimiter=',')
    
    fig, ax = plt.subplots()
    ax.tick_params(labelsize=15)
    plt.rc('legend',fontsize=18)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.get_offset_text().set_fontsize(15)
    plt.gcf().subplots_adjust(bottom=0.15)
    ax.plot(range(T),queue_mean[0], color='lightsalmon', label='SABR')
    ax.plot(range(T),queue_mean[1], '-.', color='royalblue', label='UGDA-OL')
    
    
    ax.fill_between(range(T), (queue_mean[0]-1.96*queue_sd[0]/np.sqrt(repeat)), (queue_mean[0]+1.96*queue_sd[0]/np.sqrt(repeat)), color='lightsalmon', alpha=.1 )
    ax.fill_between(range(T), (queue_mean[1]-1.96*queue_sd[1]/np.sqrt(repeat)), (queue_mean[1]+1.96*queue_sd[1]/np.sqrt(repeat)), color='royalblue', alpha=.1 )
    plt.xlabel('Time step t',fontsize=18)
    plt.ylabel('Mean queue length',fontsize=18)
    plt.legend(loc='best')
    plt.savefig('./result/'+'queue_com_'+str(load)+'_'+str(bool_)+'_plot.png',bbox_inches = "tight",dpi=300)
    plt.show()
    plt.close()
    
def plot_real_com(repeat,T,bool_):
    alg_list=['alg1','com']
    regret=np.zeros((len(alg_list),T))
    sd=np.zeros((len(alg_list),T))
    name='alg1'+'_'+'T_real'+str(T)+'bool'+str(bool_)
    regret[0]=np.loadtxt('./result/reward_mean_'+name+'.csv', delimiter=',')
    sd[0]=np.loadtxt('./result/reward_sd_'+name+'.csv', delimiter=',')
    name='com'+'_'+'T_real'+str(T)+'bool'+str(bool_)
    regret[1]=np.loadtxt('./result/reward_mean_'+name+'.csv', delimiter=',')
    sd[1]=np.loadtxt('./result/reward_sd_'+name+'.csv', delimiter=',')    
    
    fig, ax = plt.subplots()
    ax.tick_params(labelsize=15)
    plt.rc('legend',fontsize=18)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.get_offset_text().set_fontsize(15)
    plt.gcf().subplots_adjust(bottom=0.15)   
    ax.plot(range(T),regret[0], color='lightsalmon', label='SABR')
    ax.plot(range(T),regret[1],'-.', color='royalblue', label='UGDA-OL')
    ax.fill_between(range(T), (regret[0]-1.96*sd[0]/np.sqrt(repeat)), (regret[0]+1.96*sd[0]/np.sqrt(repeat)), color='lightsalmon', alpha=.1 )
    ax.fill_between(range(T), (regret[1]-1.96*sd[1]/np.sqrt(repeat)), (regret[1]+1.96*sd[1]/np.sqrt(repeat)), color='royalblue', alpha=.1 )

    plt.xlabel('Time step t',fontsize=18)
    plt.ylabel('Cumulative reward',fontsize=18)
    plt.legend(loc='upper left')
    plt.savefig('./result/'+'reward_com_real_'+str(bool_)+'_plot.png',bbox_inches = "tight",dpi=300)
    plt.show()
    plt.clf()


    queue_mean=np.zeros((len(alg_list),T))
    queue_sd=np.zeros((len(alg_list),T))
    name='alg1'+'_'+'T_real'+str(T)+'bool'+str(bool_)
    queue_mean[0]=np.loadtxt('./result/queue_mean_'+name+'.csv', delimiter=',')
    queue_sd[0]=np.loadtxt('./result/queue_sd_'+name+'.csv', delimiter=',')
    name='com'+'_'+'T_real'+str(T)+'bool'+str(bool_)
    queue_mean[1]=np.loadtxt('./result/queue_mean_'+name+'.csv', delimiter=',')
    queue_sd[1]=np.loadtxt('./result/queue_sd_'+name+'.csv', delimiter=',')
    
    fig, ax = plt.subplots()
    ax.tick_params(labelsize=15)
    plt.rc('legend',fontsize=18)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.get_offset_text().set_fontsize(15)
    plt.gcf().subplots_adjust(bottom=0.15)
    ax.plot(range(T),queue_mean[0], color='lightsalmon', label='SABR')
    ax.plot(range(T),queue_mean[1], '-.', color='royalblue', label='UGDA-OL')
    
    
    ax.fill_between(range(T), (queue_mean[0]-1.96*queue_sd[0]/np.sqrt(repeat)), (queue_mean[0]+1.96*queue_sd[0]/np.sqrt(repeat)), color='lightsalmon', alpha=.1 )
    ax.fill_between(range(T), (queue_mean[1]-1.96*queue_sd[1]/np.sqrt(repeat)), (queue_mean[1]+1.96*queue_sd[1]/np.sqrt(repeat)), color='royalblue', alpha=.1 )
    plt.xlabel('Time step t',fontsize=18)
    plt.ylabel('Mean queue length',fontsize=18)
    plt.legend(loc='best')
    plt.savefig('./result/'+'queue_com_real_'+str(bool_)+'_plot.png',bbox_inches = "tight",dpi=300)
    plt.show()
    plt.close()
    
    
def plot_queue(repeat,T,I,J,mu_inv,rho,n):
    alg_list=['alg1']
    load=rho/n

    bars=np.zeros((len(alg_list),2))
    yer=np.zeros((len(alg_list),2))
    regret=np.zeros((2,T))
    sd=np.zeros((2,T))
    tmp_regret=[]
    tmp_sd=[]
    for k,bool_ in enumerate([False,True]):
        name='alg1'+'_'+'T'+str(T)+'I'+str(I)+'J'+str(J)+'mu_inv'+str(mu_inv)+'rho'+str(rho)+'n'+str(n)+'bool'+str(bool_)
        regret[k]=np.loadtxt('./result/reward_mean_'+name+'.csv', delimiter=',')
        sd[k]=np.loadtxt('./result/reward_sd_'+name+'.csv', delimiter=',')
        
    fig, ax = plt.subplots()
    ax.tick_params(labelsize=15)
    plt.rc('legend',fontsize=18)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.get_offset_text().set_fontsize(15)
    plt.gcf().subplots_adjust(bottom=0.15)   
    ax.plot(range(T),regret[0], color='lightsalmon', label='SABR')
    ax.plot(range(T),regret[1], ':', color='mediumseagreen', label='W-SABR')

    
    
    ax.fill_between(range(T), (regret[0]-1.96*sd[0]/np.sqrt(repeat)), (regret[0]+1.96*sd[0]/np.sqrt(repeat)), color='lightsalmon', alpha=.1 )
    ax.fill_between(range(T), (regret[1]-1.96*sd[1]/np.sqrt(repeat)), (regret[1]+1.96*sd[1]/np.sqrt(repeat)), color='mediumseagreen', alpha=.1 )

    plt.xlabel('Time step t',fontsize=18)
    plt.ylabel('Regret',fontsize=18)
    plt.legend(loc='best')
    plt.savefig('./result/'+'regret_weight_'+str(load)+'_plot.png', bbox_inches = "tight",dpi=300)
    plt.show()
    plt.clf()


    bars=np.zeros((len(alg_list),2))
    yer=np.zeros((len(alg_list),2))
    queue_mean=np.zeros((2,T))
    queue_sd=np.zeros((2,T))
    for k,bool_ in enumerate([False,True]):
        name='alg1'+'_'+'T'+str(T)+'I'+str(I)+'J'+str(J)+'mu_inv'+str(mu_inv)+'rho'+str(rho)+'n'+str(n)+'bool'+str(bool_)
        queue_mean[k]=np.loadtxt('./result/queue_class_mean_'+name+'.csv', delimiter=',')
        queue_sd[k]=np.loadtxt('./result/queue_class_sd_'+name+'.csv', delimiter=',')

    fig, ax = plt.subplots()
    ax.tick_params(labelsize=15)
    plt.rc('legend',fontsize=18)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.get_offset_text().set_fontsize(15)
    plt.gcf().subplots_adjust(bottom=0.15)
    ax.plot(range(T),queue_mean[0], color='lightsalmon', label='SABR')
    ax.plot(range(T),queue_mean[1], ':', color='mediumseagreen', label='W-SABR')

    
    
    ax.fill_between(range(T), (queue_mean[0]-1.96*queue_sd[0]/np.sqrt(repeat)), (queue_mean[0]+1.96*queue_sd[0]/np.sqrt(repeat)), color='lightsalmon', alpha=.1 )
    ax.fill_between(range(T), (queue_mean[1]-1.96*queue_sd[1]/np.sqrt(repeat)), (queue_mean[1]+1.96*queue_sd[1]/np.sqrt(repeat)), color='mediumseagreen', alpha=.1 )
    plt.xlabel('Time step t',fontsize=18)
    plt.ylabel('Mean holding cost',fontsize=18)
    plt.legend(loc='best')
    plt.savefig('./result/'+'queue_weight_'+str(load)+'_plot.png', bbox_inches = "tight",dpi=300)
    plt.show()
    plt.close()

    
    bool_=False
    queue_mean=np.zeros((I,T))
    queue_sd=np.zeros((I,T))
    name='alg1'+'_'+'T'+str(T)+'I'+str(I)+'J'+str(J)+'mu_inv'+str(mu_inv)+'rho'+str(rho)+'n'+str(n)+'bool'+str(bool_)
    queue_mean=np.loadtxt('./result/queue_i_mean_'+name+'.csv', delimiter=',')
    queue_sd=np.loadtxt('./result/queue_i_sd_'+name+'.csv', delimiter=',')
    print(queue_mean.shape)

    fig, ax = plt.subplots()
    ax.tick_params(labelsize=15)
    plt.rc('legend',fontsize=18)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.get_offset_text().set_fontsize(15)
    plt.gcf().subplots_adjust(bottom=0.15)
    
    ax.plot(range(T),queue_mean[:,0], color='tomato', label=r'high priority ($c_i=7/4$)')
    ax.plot(range(T),queue_mean[:,1], color='tomato',alpha=.8)
    ax.plot(range(T),queue_mean[:,2], color='tomato',alpha=.8)
    ax.plot(range(T),queue_mean[:,3], color='tomato',alpha=.8)
    ax.plot(range(T),queue_mean[:,4], color='tomato',alpha=.8)
    
    ax.plot(range(T),queue_mean[:,5], ':', color='mediumslateblue', label=r'low priority ($c_i=1/4$)')
    ax.plot(range(T),queue_mean[:,6], ':', color='mediumslateblue',alpha=.8)
    ax.plot(range(T),queue_mean[:,7], ':', color='mediumslateblue',alpha=.8)
    ax.plot(range(T),queue_mean[:,8], ':', color='mediumslateblue',alpha=.8)
    ax.plot(range(T),queue_mean[:,9], ':', color='mediumslateblue',alpha=.8)
    plt.ylim([0, 5]) 
    plt.title('SABR',fontsize=18)
    plt.xlabel('Time step t',fontsize=18)
    plt.ylabel('Mean queue length',fontsize=18)
    plt.legend(loc='best')
    plt.savefig('./result/'+'queue_i_False'+str(load)+'_plot.png',bbox_inches = "tight",dpi=300)
    plt.show()
    plt.close()
    
    
    bool_=True
    queue_mean=np.zeros((I,T))
    queue_sd=np.zeros((I,T))
    name='alg1'+'_'+'T'+str(T)+'I'+str(I)+'J'+str(J)+'mu_inv'+str(mu_inv)+'rho'+str(rho)+'n'+str(n)+'bool'+str(bool_)
    queue_mean=np.loadtxt('./result/queue_i_mean_'+name+'.csv', delimiter=',')
    queue_sd=np.loadtxt('./result/queue_i_sd_'+name+'.csv', delimiter=',')
    fig, ax = plt.subplots()
    ax.tick_params(labelsize=15)
    plt.rc('legend',fontsize=18)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.get_offset_text().set_fontsize(15)
    plt.gcf().subplots_adjust(bottom=0.15)
    ax.plot(range(T),queue_mean[:,0], color='tomato', label=r'high priority ($c_i=7/4$)')
    ax.plot(range(T),queue_mean[:,1], color='tomato',alpha=.8)
    ax.plot(range(T),queue_mean[:,2], color='tomato',alpha=.8)
    ax.plot(range(T),queue_mean[:,3], color='tomato',alpha=.8)
    ax.plot(range(T),queue_mean[:,4], color='tomato',alpha=.8)
    
    ax.plot(range(T),queue_mean[:,5], ':', color='mediumslateblue', label=r'low priority ($c_i=1/4$)')
    ax.plot(range(T),queue_mean[:,6], ':', color='mediumslateblue',alpha=.8)
    ax.plot(range(T),queue_mean[:,7], ':', color='mediumslateblue',alpha=.8)
    ax.plot(range(T),queue_mean[:,8], ':', color='mediumslateblue',alpha=.8)
    ax.plot(range(T),queue_mean[:,9], ':', color='mediumslateblue',alpha=.8)
    plt.ylim([0, 5])

    plt.title('W-SABR',fontsize=18)
    plt.xlabel('Time step t',fontsize=18)
    plt.ylabel('Mean queue length',fontsize=18)
    plt.legend(loc='best')
    plt.savefig('./result/'+'queue_i_True'+str(load)+'_plot.png',bbox_inches = "tight",dpi=300)
    plt.show()
    plt.close()    

    
def plot_real_queue(repeat,T):
    alg_list=['alg1']

    bars=np.zeros((len(alg_list),2))
    yer=np.zeros((len(alg_list),2))
    regret=np.zeros((2,T))
    sd=np.zeros((2,T))
    tmp_regret=[]
    tmp_sd=[]
    for k,bool_ in enumerate([False,True]):
        name='alg1'+'_'+'T_real'+str(T)+'bool'+str(bool_)
        regret[k]=np.loadtxt('./result/reward_mean_'+name+'.csv', delimiter=',')
        sd[k]=np.loadtxt('./result/reward_sd_'+name+'.csv', delimiter=',')
        
    fig, ax = plt.subplots()
    ax.tick_params(labelsize=15)
    plt.rc('legend',fontsize=18)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.get_offset_text().set_fontsize(15)
    plt.gcf().subplots_adjust(bottom=0.15)   
    ax.plot(range(T),regret[0], color='lightsalmon', label='SABR')
    ax.plot(range(T),regret[1], ':', color='mediumseagreen', label='W-SABR')

    
    
    ax.fill_between(range(T), (regret[0]-1.96*sd[0]/np.sqrt(repeat)), (regret[0]+1.96*sd[0]/np.sqrt(repeat)), color='lightsalmon', alpha=.1 )
    ax.fill_between(range(T), (regret[1]-1.96*sd[1]/np.sqrt(repeat)), (regret[1]+1.96*sd[1]/np.sqrt(repeat)), color='mediumseagreen', alpha=.1 )

    plt.xlabel('Time step t',fontsize=18)
    plt.ylabel('Cumulative reward',fontsize=18)
    plt.legend(loc='best')
    plt.savefig('./result/'+'regret_real_weight'+'_plot.png', bbox_inches = "tight",dpi=300)
    plt.show()
    plt.clf()


    bars=np.zeros((len(alg_list),2))
    yer=np.zeros((len(alg_list),2))
    queue_mean=np.zeros((2,T))
    queue_sd=np.zeros((2,T))
    for k,bool_ in enumerate([False,True]):
        name='alg1'+'_'+'T_real'+str(T)+'bool'+str(bool_)
        queue_mean[k]=np.loadtxt('./result/queue_class_mean_'+name+'.csv', delimiter=',')
        queue_sd[k]=np.loadtxt('./result/queue_class_sd_'+name+'.csv', delimiter=',')

    fig, ax = plt.subplots()
    ax.tick_params(labelsize=15)
    plt.rc('legend',fontsize=18)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.get_offset_text().set_fontsize(15)
    plt.gcf().subplots_adjust(bottom=0.15)
    ax.plot(range(T),queue_mean[0], color='lightsalmon', label='SABR')
    ax.plot(range(T),queue_mean[1], '-.', color='mediumseagreen', label='W-SABR')

    
    
    ax.fill_between(range(T), (queue_mean[0]-1.96*queue_sd[0]/np.sqrt(repeat)), (queue_mean[0]+1.96*queue_sd[0]/np.sqrt(repeat)), color='lightsalmon', alpha=.1 )
    ax.fill_between(range(T), (queue_mean[1]-1.96*queue_sd[1]/np.sqrt(repeat)), (queue_mean[1]+1.96*queue_sd[1]/np.sqrt(repeat)), color='mediumseagreen', alpha=.1 )
    plt.xlabel('Time step t',fontsize=18)
    plt.ylabel('Mean holding cost',fontsize=18)
    plt.legend(loc='best')
    plt.savefig('./result/'+'queue_real_weight_'+'plot.png', bbox_inches = "tight",dpi=300)
    plt.show()
    plt.close()


    
def run_syn(I,J,d,mu_inv,T,rho,n,gamma,repeat,load,env,alg_list,bool_):    
    exp_reward=np.zeros((repeat,T))
    exp_oracle_reward=np.zeros(repeat)
    queue_length=np.zeros((repeat,T))
    queue_class_length=np.zeros((repeat,T,I))
    rho_min=rho.min()
    a1=(gamma**2)*n.sum()
    a3=((gamma+1)/(gamma-1))**2*(n.sum()**2)
    b1=I+n.sum()**2*(gamma**2/(gamma-1)**2)*((1/rho).sum())
    b3=n.sum()*gamma/(n.sum()-rho.sum())+n.sum()**3*gamma**2/((n.sum()-rho.sum())*(gamma-1))+n.sum()*gamma**2
    w=np.zeros(I)
    c=np.zeros(I)
    for i in range(I):
        if I%2==1:
            if i+1<=I/2:
                c[i]=(7/4)
            elif i+1<I:
                c[i]=(1/4)
            else:
                c[i]=1
        else: 
            if i+1<=I/2:
                c[i]=(7/4)
            else:
                c[i]=(1/4)   
                
    if bool_==True:
        for i in range(I):
            if I%2==1:
                if i+1<=I/2:
                    w[i]=(7/4)
                elif i+1<I:
                    w[i]=(1/4)
                else:
                    w[i]=1
            else: 
                if i+1<=I/2:
                    w[i]=(7/4)
                else:
                    w[i]=(1/4)   
    else:
        w=np.ones(I)
        
    rho_t_min=np.min(rho/w)

    print('oracle')
    for ind in range(repeat):
        print('repeat_ind:',ind+1)
        np.random.seed(ind+1)
        Env=SynWorld(I,J,d,mu_inv,T,rho,n,ind)  #generate environment
        oracle=Oracle(Env) #oracle algorithm
        oracle.run()
        exp_oracle_reward[ind]=oracle.exp_oracle_reward
    for alg in alg_list:
        print('alg:', alg)

        V=math.sqrt(((a3/rho_t_min+w.sum())/mu_inv)*(w.min())/a1)
        for ind in range(repeat):
            print('repeat_ind:',ind+1)
            np.random.seed(ind+1)
            Env=SynWorld(I,J,d,mu_inv,T,rho,n,ind)  #generate environment
            if alg=='alg1':
                algorithm=Algorithm1(gamma,V,T,Env,w) #scheduling algorithm
                algorithm.run()
                exp_reward[ind,:]=algorithm.exp_reward
                queue_length[ind,:]=algorithm.count_his
                queue_class_length[ind,:]=algorithm.count_class_his
                name='alg1_'+'T'+str(T)+'I'+str(I)+'J'+str(J)+'mu_inv'+str(mu_inv)+'rho'+str(round(rho.sum()))+'n'+str(round(n.sum()))+'bool'+str(bool_)
            elif alg=='com':

                algorithm=Algorithm2(gamma,V,T,Env) #algorithm for comparison
                algorithm.run()
                exp_reward[ind,:]=algorithm.exp_reward 
                queue_length[ind,:]=algorithm.count_his
                queue_class_length[ind,:]=algorithm.count_class_his
                name='com_'+'T'+str(T)+'I'+str(I)+'J'+str(J)+'mu_inv'+str(mu_inv)+'rho'+str(round(rho.sum()))+'n'+str(round(n.sum()))+'bool'+str(bool_)
        save(exp_reward,exp_oracle_reward,queue_length, queue_class_length,T,name,c)
    
def run_real(I,J,d,T,gamma,repeat,load,env,ext,prep,alg_list,K,seed,bool_):
    
    exp_oracle_reward=np.zeros(repeat)
    exp_reward=np.zeros((repeat,T))
    queue_length=np.zeros((repeat,T))
    queue_class_length=np.zeros((repeat,T,I))
    Path("./data").mkdir(parents=True, exist_ok=True)
    name='seed'+str(seed)+'_K'+str(K)
    collection=pd.read_csv('./data/pre_collection_'+name+'.csv')
    lamb=collection.groupby('cluster').size().values/T
    mu_inv=collection['N'].median()
    rho=lamb*mu_inv
    rho_min=rho.min()    
    a1=(gamma**2)
    a3=((gamma+1)/(gamma-1))**2
    b1=I+(gamma**2/(gamma-1)**2)*((1/rho).sum())
    b3=gamma/(1-rho.sum())+gamma**2/((1-rho.sum())*(gamma-1))+gamma**2
    # print(rho.sum())
    w=np.zeros(I)
    c=np.zeros(I)    
    if load==False:
        if ext==True:
            Preprocess.extraction()
        if prep==True:
            Preprocess.preprocess(K,seed)
            
    for i in range(I):
        if I%2==1:
            if i+1<=I/2:
                c[i]=(7/4)
            elif i+1<I:
                c[i]=(1/4)
            else:
                c[i]=1
        else: 
            if i+1<=I/2:
                c[i]=(7/4)
            else:
                c[i]=(1/4)   
                
    if bool_==True:
        for i in range(I):
            if I%2==1:
                if i+1<=I/2:
                    w[i]=(7/4)
                elif i+1<I:
                    w[i]=(1/4)
                else:
                    w[i]=1
            else: 
                if i+1<=I/2:
                    w[i]=(7/4)
                else:
                    w[i]=(1/4)   
    else:
        w=np.ones(I)
    rho_t_min=np.min(rho/w)
    
    print('oracle')
    for ind in range(repeat):
        print('repeat_ind:',ind+1)
        np.random.seed(ind+1)
        Env=RealWorld(I,J,d,T,ind,K,seed)   #generate environment
        oracle=Oracle(Env) #oracle algorithm
        oracle.run()
        exp_oracle_reward[ind]=oracle.exp_oracle_reward
    for alg in alg_list:
        print('alg:', alg)
        V=math.sqrt((a3/rho_t_min+w.sum())/mu_inv*(w.min())/a1)
        for ind in range(repeat):
            print('repeat_ind:',ind+1)
            np.random.seed(ind+1)
            Env=RealWorld(I,J,d,T,ind,K,seed)   #generate environment
            if alg=='alg1':
                algorithm=Algorithm1(gamma,V,T,Env,w) #scheduling algorithm
                algorithm.run()
                exp_reward[ind,:]=algorithm.exp_reward
                queue_length[ind,:]=algorithm.count_his
                queue_class_length[ind,:]=algorithm.count_class_his
                name='alg1_'+'T_real'+str(T)+'bool'+str(bool_)
            elif alg=='com':
                algorithm=Algorithm2(gamma,V,T,Env) #algorithm for comparison
                algorithm.run()
                exp_reward[ind,:]=algorithm.exp_reward 
                queue_length[ind,:]=algorithm.count_his
                queue_class_length[ind,:]=algorithm.count_class_his
                name='com_'+'T_real'+str(T)+'bool'+str(bool_)
        save(exp_reward,exp_oracle_reward,queue_length,queue_class_length,T,name,c,True)

        
def setting_syn(T, I, J, rho_tot, n_tot, mu_inv): ## set rho, n, V for syn experiment
    rho=np.zeros(I)
    n=np.zeros(J)
    for i in range(I):
        rho[i]=rho_tot/I
                
    for j in range(J):
        n[j]=max(int(n_tot/J),1)
    return rho,n

def setting_real(T,I,J,K,seed): ## set V for real experiment
    name='seed'+str(seed)+'_K'+str(K)
    collection=pd.read_csv('./data/pre_collection_'+name+'.csv')
    lamb=collection.groupby('cluster').size().values/T
    mu_inv=collection['N'].median()
    rho=lamb*mu_inv
    rho_tot=rho.sum()
    rho_harm=len(rho)/np.sum(1.0/rho) 
    rho_min=rho.min()
    rho_max=rho.max()

    V_1=math.sqrt(I*T*(1+1/rho_harm)/mu_inv)
    V_2=math.sqrt((rho_min*rho_tot+rho_max**2/rho_min)*T/mu_inv)
    return V_1, V_2
    
if __name__ == "__main__":   
    opt = str(sys.argv[1]) #input1; '1': figure 3, '2': figure 4, '3': figure 5, '4': figure 6
    mode = str(sys.argv[2]) #input2; 'load' : load data
    repeat=10
    d=2 # dimension for context vectors
    gamma=1.2 # gamma>1
    if mode=='load':
        load=True
        print('load data')
    elif mode=='run':
        load=False
        
    if opt=='1': 
        T=500
        bool_=False
        env='SynWorld'
        mu_inv=1 # mean job processing time
        rho_tot=1 #total arrival rate for processing time
        n_tot=4  #total departure rate for processing time
        I=10 # number of job classes
        J=2  # number of servers
        alg_list=['alg1','com']
        rho,n=setting_syn(T,I,J,rho_tot,n_tot,mu_inv)
        if load==False:
            run_syn(I,J,d,mu_inv,T,rho,n,gamma,repeat,load,env,alg_list,bool_)    
        plot_com(repeat,T,I,J,mu_inv,rho_tot,n_tot,bool_)
    
    
    elif opt=='2': 
        T=500
        bool_=True
        env='SynWorld'
        mu_inv=1 # mean job processing time
        rho_tot=1 #total arrival rate for processing time
        n_tot=4  #total departure rate for processing time
        I=10 # number of job classes
        J=2  # number of servers
        alg_list=['alg1']
        rho,n=setting_syn(T,I,J,rho_tot,n_tot,mu_inv)
        if load==False:
            bool_=True
            run_syn(I,J,d,mu_inv,T,rho,n,gamma,repeat,load,env,alg_list,bool_)   
            bool_=False
            run_syn(I,J,d,mu_inv,T,rho,n,gamma,repeat,load,env,alg_list,bool_)   
        plot_queue(repeat,T,I,J,mu_inv,rho_tot,n_tot)
    


    elif opt=='3': #exp with real data
        env='RealWorld'
        bool_=False
        J=12
        d=4
        repeat=10
        T=1100
        K=5
        I=K
        seed=26
        gamma=1.2
        ext=True #True: extract real data
        prep=True #True: preprocess real data
        alg_list=['alg1','com']
        if load==False:
            run_real(I,J,d,T,gamma,repeat,load,env,ext,prep,alg_list,K,seed,bool_)
        plot_real_com(repeat,T,bool_)  
        
    elif opt=='4': #exp with real data
        env='RealWorld'
        bool_=True
        J=12
        d=4
        repeat=10
        T=1100
        K=5
        I=K
        seed=26
        gamma=1.2
        ext=True #True: extract real data
        prep=True #True: preprocess real data
        alg_list=['alg1']
        if load==False:
            bool_=True
            run_real(I,J,d,T,gamma,repeat,load,env,ext,prep,alg_list,K,seed,bool_)
            bool_=False
            run_real(I,J,d,T,gamma,repeat,load,env,ext,prep,alg_list,K,seed,bool_)
        plot_real_queue(repeat,T)         
    
    
    else:
        print('wrong input')
