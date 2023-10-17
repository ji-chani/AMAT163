## Solving Knapsack Problem using Hill Climbing Algorithm

# importing packages
import numpy as np
import copy

# Parameters
numVar = 8
labels = ['wine', 'beer', 'pizza', 'burger', 'fries', 'coke', 'apple', 'donut']
values = np.array([89, 90, 30, 50, 90, 79, 90, 10])
weights = np.array([123, 154, 258, 354, 365, 150, 95, 195])
weight_limit = 750
max_iteration = 50
minimization = True
method = 'Random' # try 'Greedy Search'

# ---- Helper Functions (do not change)
def get_value(soln, values=values):
    return np.matmul(soln, values)

def get_weight(soln, weights=weights):
    return np.matmul(soln, weights)

def get_order(minimization):
    if minimization:
        return 1
    else:
        return -1

def generate_init_soln(size, values=values, weights=weights, limit=weight_limit, minimization=minimization, method=method):
    """
    Parameter
    -----------
    size: Number of decision variables
    method: 'Random' or 'Greedy Search'

    Returns
    -----------
    `np.ndarray` initial solution.

    """

    order = get_order(minimization)

    if method == "Random":
        feasibility = False
        while feasibility == False:
            initial_soln = np.random.randint(0, 2, size)
            w = get_weight(initial_soln, weights)
            if order*w >= order*limit:
                feasibility = True

    if method == 'Greedy Search':
        indices = values.argsort()[::order]
        initial_soln = np.zeros(size)
        new_init_soln = initial_soln

        if minimization:
            for ind in range(len(indices)):
                new_init_soln = copy.deepcopy(new_init_soln)
                new_init_soln[indices[ind]] = 1
                w = get_weight(new_init_soln, weights)
                if order*w >= order*limit:
                    break
                else:
                    initial_soln[indices[ind]] = 1
            initial_soln[indices[ind]] = 1
        else:
            for ind in range(len(indices)):
                new_init_soln = copy.deepcopy(new_init_soln)
                new_init_soln[indices[ind]] = 1
                w = get_weight(new_init_soln, weights)
                if order*w >= order*limit:
                    initial_soln[indices[ind]] = 1
                else:
                    break

    return initial_soln

def generate_neighbors(soln):
    neighbors = []
    for i in range(len(soln)):
        new_neighbor = copy.deepcopy(soln)
        if new_neighbor[i] == 1:
            new_neighbor[i] = 0
        else:
            new_neighbor[i] = 1
        neighbors.append(new_neighbor)
    return neighbors

def select_best_neighbor(neighbors, values=values, weights=weights, limit=weight_limit, minimization=minimization):
    """
    Selects the "best" feasible neighbor from a set of neighbors

    Parameter
    ------------
    neighbors

    Return
    -----------
    best_nb: single neighbor
    """
    order = get_order(minimization)
    nb_weights = get_weight(neighbors)
    filtered_inds = np.where(order*nb_weights >= order*limit)[0]
    feasible_nb = [neighbors[i] for i in filtered_inds]

    if minimization:
        best_ind = np.argmin(get_value(feasible_nb))
    else:
        best_ind = np.argmax(get_value(feasible_nb))

    return feasible_nb[best_ind]

def compare_solns(incumbent_soln, best_nb, minimization=minimization, continue_iter=True):
    order = get_order(minimization)
    prev_value, new_value = get_value(incumbent_soln), get_value(best_nb)

    if order*prev_value <= order*new_value:
        continue_iter = False
    return continue_iter

def optimal_soln(incumbent_soln, labels=labels):
    opt_labels = []
    for i in range(len(incumbent_soln)):
        if incumbent_soln[i] == 1:
            opt_labels.append(labels[i])
    return opt_labels


# ------ Implementation
continue_iter = True
iter_count = 0

init_soln = generate_init_soln(size=numVar)
print('---------------------------')
print(f'Initial solution: {init_soln} \n value = {get_value(init_soln)}, weight = {get_weight(init_soln)} \n')

incumbent_soln = init_soln

while continue_iter:
    print(f'ITERATION {iter_count+1}: --------- \n')

    # generating neighbors
    neighbors = generate_neighbors(incumbent_soln)
    print(f'Neighbors: \n values = {get_value(neighbors)} \n weights = {get_weight(neighbors)} \n')

    # selecting best neighbor
    best_nb = select_best_neighbor(neighbors)
    print(f'Best neighbor: {best_nb} \n value = {get_value(best_nb)} \n weight = {get_weight(best_nb)} \n')

    # compare incumbent solution and best neighbor
    continue_iter = compare_solns(incumbent_soln, best_nb)
    if continue_iter:
        incumbent_soln = best_nb

    if iter_count == max_iteration:
        break

    print(f'Incumbent solution: {incumbent_soln} \n value = {get_value(incumbent_soln)} \n weight = {get_weight(incumbent_soln)} \n')
    iter_count += 1

print('--------------------------------------------')
print(f'After {iter_count} iteration(s), optimal solution is {incumbent_soln} or to include \n {optimal_soln(incumbent_soln, labels)} \n value = {get_value(incumbent_soln)} \n weight = {get_weight(incumbent_soln)} \n')