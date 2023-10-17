## Solving TSP using Hill Climbing Algorithm

# importing packages
import numpy as np
import random

# ---- Parameters
cost_matrix = [
    [0, 1559, 921, 1334],
    [1559, 0, 809, 1397],
    [921, 809, 0, 921],
    [1334, 1397, 921, 0]
]  # in hundred pesos
nodeLabels = ["Home", "A", "B", "C"]
cost_limit = 5000 # in hundred pesos
minimization = True
maxIteration = 10

# Helper Functions
def get_order(minimization):
    if minimization:
        return 1
    else:
        return -1

def adjust_costMatrix(cost_matrix, minimization):
    order = get_order(minimization)
    const = order * np.inf
    for i in range(len(cost_matrix)):
        cost_matrix[i][i] = const
    return cost_matrix

def generate_init_soln(cost_matrix, minimization, method='Random'):
    initSoln = []
    if method == 'Random':
        cities = list(range(len(cost_matrix)))
        for i in range(len(cities)):
            randomCity = cities[random.randint(0, len(cities)-1)]
            initSoln.append(randomCity)
            cities.remove(randomCity)
        return initSoln
    
    if method == 'Greedy Search':
        order = get_order(minimization)

        return initSoln

def routeCost(cost_matrix, solution):
    routeCost = 0
    for i in range(len(solution)):
        routeCost += cost_matrix[solution[i-1]][solution[i]]
    return routeCost

def getNeighbors(solution, method='2-edge'):
    neighbors = []

    if method == 'swap':
        for i in range(len(solution)):
            for j in range(i+1, len(solution)):
                nb = solution.copy()
                nb[i] = solution[j]
                nb[j] = solution[i]
                neighbors.append(nb)
    if method == '2-edge':
        for i in range(len(solution)):
            for j in range(i+1, len(solution)):
                nb = []
                for k in range(0, i):
                    nb.append(solution[k])
                nb.append(solution[j])
                for k in range(j-1, i, -1):
                    nb.append(solution[k])
                nb.append(solution[i])
                for k in range(j+1, len(solution)):
                    nb.append(solution[k])
                    
                neighbors.append(nb)
    return neighbors

def neighborCost(neighbors, cost_matrix):
    neighborCost = []
    for i in range(len(neighbors)):
        neighborCost.append(routeCost(cost_matrix, neighbors[i]))
    return neighborCost

def selectBestNeighbor(neighbors, nbCosts, cost_matrix, cost_limit, minimization): # best improvement
    order = get_order(minimization)
    filtered_inds = np.where(order*np.array(nbCosts) <= order*cost_limit)[0]
    feasible_nb = [neighbors[i] for i in filtered_inds]
    sorted_inds = np.array(neighborCost(feasible_nb, cost_matrix)).argsort()[::order]
    return feasible_nb[sorted_inds[0]]

def compareSolns(prev_value, new_value, minimization, continue_iter=True):
    order = get_order(minimization)

    if order*prev_value <= order*new_value:
        continue_iter = False
    return continue_iter

def optimalSoln(solution, nodeLabels, startingNode):
    optSoln = []
    ind = np.where(np.array(solution) == startingNode)[0][0]
    for i in range(ind, len(solution)):
        optSoln.append(i)
    for i in range(ind):
        optSoln.append(i)
    optimal_ind = [solution[i] for i in optSoln]
    optimal_ind.append(optimal_ind[0])
    return [nodeLabels[k] for k in optimal_ind]

# adjust cost matrix (changes the diagonal of matrix depending on objective)
cost_matrix = adjust_costMatrix(cost_matrix, minimization)

# -------- Implementation

# initial solution
method = 'Random'
initSoln = generate_init_soln(cost_matrix, minimization, method)
print(f'Initial Solution: {initSoln} \n Route Cost = {routeCost(cost_matrix, initSoln)} \n')

incumbentSoln = initSoln

continue_iter = True
iter_count = 0
while continue_iter:
    print(f'Iteration {iter_count+1} -------------- \n')

    # cost of current incumbent solution
    prev_cost = routeCost(cost_matrix, incumbentSoln)

    # generate neighbors (2-edge exchange)
    neighbors = getNeighbors(incumbentSoln, method='2-edge')
    nbCosts = [routeCost(cost_matrix, neighbors[i]) for i in range(len(neighbors))]
    print(f'Neighbors: {neighbors}')
    print(f'Neighbor Route Costs = {nbCosts} \n')

    # select best neighbor (best improvement)
    best_nb = selectBestNeighbor(neighbors, nbCosts, cost_matrix, cost_limit, minimization)
    new_cost = routeCost(cost_matrix, best_nb)
    print(f'Best Neighbor : {best_nb} \n Route Cost = {new_cost}')

    # compare solutions
    continue_iter = compareSolns(prev_cost, new_cost, minimization)
    if continue_iter:
        incumbentSoln = best_nb

    if iter_count == maxIteration:
        break

    print(f'Incumbent Solution: {incumbentSoln} \n Route Cost: {routeCost(cost_matrix, incumbentSoln)} \n')
    iter_count += 1

print('--------------------------------------------')
print(f'After {iter_count} iteration(s), optimal solution is {incumbentSoln} \n Route Cost = {routeCost(cost_matrix, incumbentSoln) * 100}')

print(f'That is, optimal solution is to travel the path \n {optimalSoln(incumbentSoln, nodeLabels, startingNode=0)} \n')