# Scheduling_BilinearRewards


## Requirement
 Python 3 >=3.5

## Structure
  * main.py\
    This file includes the main function.

  * Preprocess.py\
  This file includes the code for extracting and preprocessing real data. It is required to put your own google cloud key in this file to extract the public dataset described in https://github.com/google/cluster-data. Otherwise, you can use the dataset in the 'data' file extracted from the public dataset by deactivating extraction in main.py (i.e. ext=False). 

  * Environment.py\
  This file includes the code for generating an environment (synthetic world or real world) of a queueing system with the bilinear reward structure. 
  
  * Algorithm.py\
  This file includes the code for scheduling algorithms.

  * Oracle.py\
  This file includes the code for running the oracle policy.

## How to run this code
Please run this command:

Figure 3:
```python3 main.py 1 run```

Figure 4: 
```python3 main.py 2 run```

Figure 5: 
```python3 main.py 3 run```

Figure 6: 
```python3 main.py 4 run```

