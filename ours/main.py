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
    bounds = [(0, None) for _ in range(I * J)]

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
    SEED = 777
    np.random.seed(SEED)

    ## Initialize necessary values
    V = 1       # Placeholder value
    I = 20     # total number of jobs (patients)
    J = 10      # total number of servers (hospitals)
    lbdas = sample_lambdas(I, J)
    T = 100    # total horizon
    true_mean_reward = np.random.uniform(low=0, high=1, size=J)
    min_reward = np.amin(true_mean_reward)

    ## Initialize matrices
    Gamma = np.zeros((I, J))            # Gamma[i][j]: the number of patients of type i arrived at the hospital j
    Q = np.zeros((I, J))                # Q[i][j]: the number of waiting patients of type i in the hospital j
    B = np.zeros((I, J))                # B[i][j]: the number of treated patients of type i in the hospital j
    H = np.zeros((I, J))                # H_{ij}(t) = \sum_{\tau=1}^t B_{ij}(\tau)
    # R = np.zeros((I, J))              # matrix for the rewards
    R_bar = np.zeros((I, J))            # matrix for the average rewards
    queue = {j: [] for j in range(J)}   # queue for each hospital; contain the type of waiting patients
    
    ## run simulator
    ## 1. for each time
    for t in tqdm(range(T)):
        ## 2. implement the Step 1
        R = np.zeros((I, J))    # matrix containing r_{ij}(t)
        for i in range(I):
            for j in range(J):
                if H[i][j] == 0:
                    R[i][j] = 1
                else:
                    min_inside = R_bar[i][j] + np.sqrt((2 * np.log(t)) / H[i][j])
                    minimum = np.minimum(min_inside, 1)
                    R[i][j] = np.maximum(minimum, min_reward)        
        
        ## 3. implement the Step 2
        p = optimize(V=V, I=I, J=J, lbdas=lbdas, R=R)
        # print(p, np.sum(p, axis=1))

        ## 4. Step 3 Part 1 - with returned r_{ij}(t) and p_{ij}(t) implement update
        arrivals = np.random.poisson(lam=lbdas)
        for i in range(I):
            arrival_num = arrivals[i]
            for _ in range(arrival_num):
                to_assign = np.random.choice(J, replace=True, p=p[i, :])
                Gamma[i][to_assign] += 1
                queue[to_assign].append(i)
                Q[i][to_assign] += 1 ## 9/3 add
        
        # # update B and Q
        # for i in range(I):
        #     for j in range(J):
        #         B[i][j] = np.minimum(Q[i][j] + Gamma[i][j], 1)
        #         Q[i][j] = np.minimum(Q[i][j] + Gamma[i][j] - 1, 0)


        ## add 9/3
        B = np.zeros((I, J))
        for j in range(J):
            for i in range(I):
                if len(queue[j]) == 0:
                    continue
                elif queue[j][0] == i:
                    B[i][j] = 1
                    Q[i][j] = max(Q[i][j] - 1, 0) ## or Q[i][j] -= 1


        ## 5. Step 3 Part 2
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
    sns.heatmap(p, annot=False, cmap='viridis')

    # Add labels and title
    plt.title(r"Heatmap of $p$")
    plt.xlabel('Server (Hospitals)')
    plt.ylabel('Jobs (Patients)')
    filename = "./figures/p_no_annotation.pdf"
    plt.savefig(filename)

    plt.show()
