## Solving an Unconstrained Nonlinear Optimization Problem

# importing packages
import numpy as np

# ------------ Parameters
delta = 0.1  # small incremenets in x and y
minimization = True
max_iterations = 100

# Helper Functions (do not change)
def func(x, y):
    return x**2 + y**2 - 6*x + 4*y

def generate_init_soln():
    return np.random.randn(1)[0], np.random.randn(1)[0]

def generate_neighbors(x, y, delta=delta):
    X = [x, x+delta, x-delta]
    Y = [y, y+delta, y-delta]
    nb = [(x,y) for x in X for y in Y]
    nb.remove((x, y))
    return nb

def get_order(minimization):
    if minimization:
        return 1
    else:
        return -1
    
def next_solution(x, y, minimization=minimization):
    order = get_order(minimization)
    neighbors = generate_neighbors(x, y)
    values = np.array([func(nb[0], nb[1]) for nb in neighbors])
    sorted_inds = values.argsort()[::order]

    return neighbors[sorted_inds[0]], values[sorted_inds[0]]

def compare_solns(prev_value, new_value, iter_count, max_iterations=max_iterations, continue_iter=True, minimization=minimization):
    order = get_order(minimization)
    if order*prev_value <= order*new_value or iter_count == max_iterations:
        continue_iter = False
    return continue_iter


# ------ Implementation
continue_iter = True
iter_count = 0

x0, y0 = generate_init_soln()
print(f'Initial Solution: ({x0:.2f},{y0:.2f}) value = {func(x0,y0):.4f}')

incumbent_soln = (x0, y0)
while continue_iter:
    print(f'ITERATION {iter_count+1}: -------------- \n')
    x, y = incumbent_soln[0], incumbent_soln[1]
    prev_value = func(x, y)

    # generate neighbors and selecting best neighbor
    new_soln, new_value = next_solution(x, y)
    print(f'Best Neighbor: ({new_soln[0]:.2f},{new_soln[1]:.2f}), value = {new_value:.4f}')

    # compare incumbent solution and best neighbor
    continue_iter = compare_solns(prev_value, new_value, iter_count)
    if continue_iter:
        incumbent_soln = new_soln
    
    if iter_count == max_iterations:
        break
    
    print(f'Incumbent Solution: ({incumbent_soln[0]:.2f}, {incumbent_soln[1]:.2f}), value = {func(incumbent_soln[0], incumbent_soln[1]):.4f} \n')
    iter_count += 1

print('---------------------------------------')
print(f'Optimal Solution after {iter_count} iteration is ({incumbent_soln[0]:.2f}, {incumbent_soln[1]:.2f}), value = {func(incumbent_soln[0], incumbent_soln[1]):.2f} \n')