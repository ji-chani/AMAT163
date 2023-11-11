# Solving Nonlinear Optimization Problem using Brute Force Enumeration

# packages
import numpy as np
from itertools import product

# Helper Functions
def func(soln: list):
    x, y, z = soln[0], soln[1], soln[2]
    return x**2 + y**2 + z**2 - 6*x + 4*y - 2*z

def memoryValues(harmonyMemory: list):
    return [func(soln) for soln in harmonyMemory]


# Main Function
def bruteForce_NonLinear(numVars, interval, minimization):
    # listing all possible solutions
    possibleValues = list(range(interval[0], interval[1]+1))
    allSolutions = [p for p in product(possibleValues, repeat=numVars)]

    # sorting values of solutions
    if minimization:
        order = 1
    else:
        order = -1
    values = memoryValues(allSolutions)
    sortedValuesInds = np.array(values).argsort()[::order]
    optimal = allSolutions[sortedValuesInds[0]]

    print(f'Using Brute Force Enumeration')
    print(f'Optimal Solution: \n {optimal}, Value: {func(optimal)}')

# Parameters
numVars = 3
interval = (-10, 10)
minimization = True

# Main Implementation
bruteForce_NonLinear(numVars, interval, minimization)