"""
Algorithm 1: Barrier-Function-Based Algorithm
- treat과 wait은 학습에 사용되지 않음
- reward 하한인 r_* (min_reward)는 J개 reward를 uniform random sampling하고 이를 true_mean_reward에 할당 한 뒤, 그 중 최소값을 r_* (min_reward)로 설정
- lambda의 합이 J보다 작도록 조정하기 위해 epsilon을 추가함
- 다만 lambda의 범위가 1~I로 설정되어 있는데, I이어야 할 필요는 없을 수 있음. 논문에서는 job(I) 각각 2,3으로 뒀음
- plot 여러 개 고려: cumulative rewards, waiting time, number of treated patients

"""
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


def sample_lambdas(high:int, size:int, epsilon:float=0.1) -> np.ndarray[float]:
    """
    high: the number of parameters to be sampled; set as the total number of patients
    size: upper bound of the sum of total parameters; set as the total number of servers
    epsilon: adjustment constant to make the sum of the samples less than the size
    """
    lambdas = np.random.randint(low=1, high=size, size=high)
    lambdas = lambdas / (np.sum(lambdas) + epsilon) # to make np.sum(lambdas) < size
    lambdas = lambdas * size 
    return lambdas

def optimize(V:float, I:int, J:int, lbdas:np.ndarray, R:np.ndarray):
    ## Objective function to be maximized (negating for minimization)
    def objective(p:np.ndarray):
        p = np.reshape(p, (I, J))
        term1 = np.log(1 - np.sum(lbdas[:, None] * p, axis=0))
        term2 = np.sum(lbdas[:, None] * p * R, axis=0)
        return -np.sum((1/V) * term1 + term2)
    
    ## Constraints
    def constraint_sum(p):
        p = np.reshape(p, (I, J))
        return np.sum(p, axis=1) - 1
    
    # Bounds for each p_ij(t)
    bounds = [(0, 1) for _ in range(I * J)]

    # Initial guess for p_ij(t)
    p0 = np.zeros((I, J)).flatten()

    # Define the constraints in the form required by 'minimize'
    constraints = [{'type': 'eq', 'fun': constraint_sum}]

    # Perform the optimization
    result = minimize(objective, p0, bounds=bounds, constraints=constraints)

    # Reshape the result to match p_ij(t)
    p_optimized = np.reshape(result.x, (I, J))

    return p_optimized


if __name__ == "__main__":
    ## set the seed
    SEED = 145
    np.random.seed(SEED)

    ## Initialize necessary values
    V = 1      # Placeholder value
    I = 2      # total types of jobs (patients)
    J = 6      # total types of servers (hospitals)
    # lbdas = sample_lambdas(I, J)
    lbdas = np.array([2, 3])
    T = 1000    # total horizon
    true_mean_reward = np.random.uniform(low=0, high=1, size=J)
    min_reward = np.amin(true_mean_reward)
    initial_reward = 1

    ## Initialize matrices
    Gamma = np.zeros((I, J))            # Gamma[i][j]: the number of patients of type i arrived at the hospital j
    wait = np.zeros((I, J))             # wait[i][j]: the number of waiting patients of type i in the hospital j
    treat = np.zeros((I, J))            # treat[i][j]: the number of treated patients of type i in the hospital j
    H = np.zeros((I, J))                # H_{ij}(t) = \sum_{\tau=1}^t treat_{ij}(\tau)
    # R = np.zeros((I, J))              # matrix for the rewards
    R_bar = np.zeros((I, J))            # matrix for the average rewards
    queue = {j: [] for j in range(J)}   # queue for each hospital; contain the type of waiting patients
    U = np.zeros((I, J))
    
    ## run simulator
    ## 1. for each time
    for t in range(1,T+1):
    # for t in tqdm(range(1,T+1)):
        ## 2. implement the Step 1
        R = np.zeros((I, J))    # matrix containing r_{ij}(t)
        R[:, :] = 0
        # print(f"min_reward: {min_reward}")
        print(f"time:\t{t}, R_bar:\n", R_bar)
        for i in range(I):
            for j in range(J):
                if H[i][j] == 0:
                    R[i][j] = initial_reward ## 1->0 10/22 for test
                else:
                    uncertainty = np.sqrt((np.log(t-1)) / H[i][j])
                    min_inside = R_bar[i][j] + uncertainty
                    minimum = np.minimum(min_inside, 1)
                    R[i][j] = np.maximum(minimum, min_reward)  
                    print(f"time = {t}\tH_ij = {H[i][j]}\tr_ij = {R[i][j]}\trbar_ij = {R_bar[i][j]}")
        
        ## 3. implement the Step 2
        p = optimize(V=V, I=I, J=J, lbdas=lbdas, R=R)
        print(np.sum(p, axis=1))

        ## 4. Step 3 Part 1 - with returned r_{ij}(t) and p_{ij}(t) implement update
        arrivals = np.random.poisson(lam=lbdas)
        for i in range(I):
            arrival_num = arrivals[i]
            for _ in range(arrival_num):
                to_assign = np.random.choice(J, replace=True, p=p[i, :])
                Gamma[i][to_assign] += 1
                queue[to_assign].append(i)
                wait[i][to_assign] += 1 ## 9/3 add
        
        # # update treat and wait
        # for i in range(I):
        #     for j in range(J):
        #         treat[i][j] = np.minimum(wait[i][j] + Gamma[i][j], 1)
        #         wait[i][j] = np.minimum(wait[i][j] + Gamma[i][j] - 1, 0)


        ## add 9/3 
        ## update treat and wait
        treat = np.zeros((I, J))
        for j in range(J):
            for i in range(I):
                if len(queue[j]) == 0:
                    continue
                elif queue[j][0] == i:
                    treat[i][j] = 1
                    wait[i][j] = max(wait[i][j] - 1, 0) ## or wait[i][j] -= 1


        ## 5. Step 3 Part 2
        ## update H, R_bar, and queue
        for j in range(J):
            # if the queue at server j is not empty
            if len(queue[j]) > 0:
                # observe the type of the first job in the queue
                i_star = queue[j][0]
                # observe the realized reward
                r_ij = R[i_star][j]
                # print(f"time: {t}\tr_ij = {r_ij}")
                X_ij = np.random.binomial(n=1, p=r_ij)

                # Update
                h_prev = H[i_star][j]           # h_ij(t-1)
                H[i_star][j] += 1               # h_ij(t)
                r_bar_prev = R_bar[i_star][j]   # r_bar(t-1)
                R_bar[i_star][j] = (h_prev * r_bar_prev + X_ij) / H[i_star][j]
                queue[j].pop(0) ## add 9/3
        
    print("Done!")

    # Create the heatmap
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    sns.heatmap(p, annot=False, ax=ax[0], cmap='viridis')
    ax[0].set_title(r"Heatmap of $p$")
    ax[0].set_xlabel('Server (Hospitals)')
    ax[0].set_ylabel('Jobs (Patients)')

    sns.heatmap(H, annot=False, ax=ax[1], cmap='plasma')
    ax[1].set_title(r"Heatmap of $H$")
    ax[1].set_xlabel('Server (Hospitals)')
    ax[1].set_ylabel('Jobs (Patients)')

    plt.tight_layout()
    filename = f"./figures/p_H_no_annotation_seed_{SEED}_initial_reward_{initial_reward}_min_sampled.pdf"
    plt.savefig(filename)

    plt.show()

    print(f"min reward : {min_reward}")