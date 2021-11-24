# Scheduling_BilinearRewards


## Requirement
 Python 3 >=3.5

## Structure
  * main.py\
    This file includes the main function.
    * For getting the results in Figure 1, please set variables for synthetic data in the main function as follows:\
   I=10/J=2/T=700/d=2/mu_inv=1/rho_tot=1/n_tot=8/gamma=1.2/repeat=10/util_arriv=False/load=False/com=True/fix=False 
    * For getting the results in Figure 7, please set variables for real data in the main function as follows\
   I=5/J=12/d=4/T=1100/gamma=1.2/repeat=10/ext=False/prep=False/load=False/com=True

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
