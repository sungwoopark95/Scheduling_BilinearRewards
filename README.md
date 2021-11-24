# Scheduling_BilinearRewards


## Requirement
 Python 3 >=3.5

## Structure
  * main.py\
  This file includes the main function.
  ** For getting the results in Figure 1, set variables for synthetic data in the main function as follows:\
  I=10 # number of job classes\
  J=2  # number of servers\
  T=700 # time horizon\
  d=2 # dimension for context vectors\
  mu_inv=1 # mean job processing time\
  rho_tot=1  #total arrival rate for processing time\
  n_tot=8  #total departure rate for processing time\
  gamma=1.2 # gamma>1\
  repeat=10  #repeat number\
  util_arriv=False #True: utilize traffic intensities for Algorithm 1\
  load=False #True: load saved data without running the algorithm\
  com=True #True: compare with other algorithm\
  fix=False #True: alternative setting with fixed job arrival rates and server capacities \
  
  2. For getting the results in Figure 7, set variables for real data in the main function as follows
  I=5\
  J=12\
  d=4\
  T=1100\
  gamma=1.2\
  repeat=10\
  ext=False #True: extract real data\
  prep=False #True: preprocess real data\
  load=False\
  com=True\

  * Preprocess.py\
  This file includes the code for extracting and preprocessing real data. It is required to put your own google cloud key in this file to extract the public dataset. Otherwise, you can use the dataset in the 'data' file by deactivating extraction in main.py (ext=False).

  * Environment.py\
  This file includes the code for generating an environment (synthetic world or real world) of a queueing system with the bilinear reward structure. 
  
  * Algorithm.py\
  This file includes the code for scheduling algorithms.

  * Oracle.py\
  This file includes the code for running the oracle policy.

## How to run this code
Please run this command:

 * Synthetic data\
 python3 main.py syn

 * Real data\
 python3 main.py real
