"""
Algorithm 2: Queue-Based Algorithm

"""
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


def sample_lambdas(high:int, size:int, epsilon:float=0.1) -> np.ndarray[float]:
    """
    high: upper bound of samples; set as the total number of patients
    size: the number of parameters to be sampled; set as the total number of servers
    epsilon: adjustment constant to make the sum of the samples less than the size
    """
    lambdas = np.random.randint(low=1, high=high, size=high)
    lambdas = lambdas / (np.sum(lambdas) + epsilon) # to make np.sum(lambdas) < size
    lambdas = lambdas * size 
    return lambdas


if __name__ == "__main__":
    ## set the seed
    SEED = 145
    np.random.seed(SEED)

    ## Initialize necessary values
    V = 1000       # Placeholder value
    I = 2     # total number of jobs (patients)
    J = 6      # total number of servers (hospitals)
    # lbdas = sample_lambdas(I, J)
    lbdas = np.array([2, 3])
    T = 1000    # total horizon
    true_mean_reward = np.random.uniform(low=0, high=1, size=J)
    min_reward = 0.1
    epsilon = 1 / np.sqrt(T)
    initial_reward = 0.5

    ## Initialize matrices
    Gamma = np.zeros((I, J))            # Gamma[i][j]: the number of patients of type i arrived at the hospital j
    wait = np.zeros((I, J))             # wait[i][j]: the number of waiting patients of type i in the hospital j
    treat = np.zeros((I, J))            # treat[i][j]: the number of treated patients of type i in the hospital j
    H = np.zeros((I, J))                # H_{ij}(t) = \sum_{\tau=1}^t treat_{ij}(\tau)
    # R = np.zeros((I, J))              # matrix for the rewards
    R_bar = np.zeros((I, J))            # matrix for the average rewards
    queue = {j: [] for j in range(J)}   # queue for each hospital; contain the type of waiting patients
    
    ## run simulator
    ## 1. for each time
    for t in range(1,T+1):
        ## 2. implement the Step 1
        R = np.zeros((I, J))    # matrix containing r_{ij}(t)
        for i in range(I):
            for j in range(J):
                if H[i][j] == 0:
                    R[i][j] = initial_reward
                else:
                    min_inside = R_bar[i][j] + np.sqrt((np.log(t-1)) / H[i][j])
                    minimum = np.minimum(min_inside, 1)
                    R[i][j] = np.maximum(minimum, min_reward)
                    print(f"time = {t}\tH_ij = {H[i][j]}\tr_ij = {R[i][j]}\trbar_ij = {R_bar[i][j]}")
        
        ## 3. Step 3 Part 1 - with returned r_{ij}(t) and p_{ij}(t) implement update
        arrivals = np.random.poisson(lam=lbdas)
        for i in range(I):
            arrival_num = arrivals[i]
            for _ in range(arrival_num):
                to_assign = np.argmax(R[i] - epsilon * wait[i]) ## np.argmax returns the first index of the maximum value
                Gamma[i][to_assign] += 1
                queue[to_assign].append(i)
                wait[i][to_assign] += 1 ## 9/3 add
        
        # # update treat and wait
        # for i in range(I):
        #     for j in range(J):
        #         treat[i][j] = np.minimum(wait[i][j] + Gamma[i][j], 1)
        #         wait[i][j] = np.minimum(wait[i][j] + Gamma[i][j] - 1, 0)


        ## add 9/3
        ## treatment starts 
        ## update treat and wait
        treat = np.zeros((I, J))
        for j in range(J):
            for i in range(I):
                if len(queue[j]) == 0:
                    continue
                elif queue[j][0] == i:
                    treat[i][j] = 1
                    wait[i][j] = max(wait[i][j] - 1, 0) ## or wait[i][j] -= 1


        ## 4. Step 3 Part 2
        ## update H, R_bar, and queue
        for j in range(J):
            # if the queue at server j is not empty
            if len(queue[j]) > 0:
                # observe the type of the first job in the queue
                i_star = queue[j][0]
                # observe the realized reward
                r_ij = R[i_star][j]
                X_ij = np.random.binomial(n=1, p=r_ij)

                # Update
                h_prev = H[i_star][j]           # h_ij(t-1)
                H[i_star][j] += 1               # h_ij(t)
                r_bar_prev = R_bar[i_star][j]   # r_bar(t-1)
                R_bar[i_star][j] = (h_prev * r_bar_prev + X_ij) / H[i_star][j]
                queue[j].pop(0) ## add 9/3
    print("Done!")

    # Create the heatmap
    sns.heatmap(H, annot=False, cmap='plasma')
    plt.title(r"Heatmap of $H$")
    plt.xlabel('Server (Hospitals)')
    plt.ylabel('Jobs (Patients)')
    filename = f"./figures/H_no_annotation_seed_{SEED}_initial_reward_{initial_reward}_epsilon_{epsilon}.pdf"
    plt.savefig(filename)

    plt.show()
