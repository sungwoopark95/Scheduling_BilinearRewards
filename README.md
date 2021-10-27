# Scheduling_BilinearRewards


## Requirement
 Python 3 >=3.5

## Structure
  * main.py\
  This file includes the main function.

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
