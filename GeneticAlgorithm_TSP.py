import numpy as np
import copy

def route_cost(solution, costMatrix):
  routeCost = 0
  for i in range(len(solution)):
      routeCost += costMatrix[solution[i-1]][solution[i]]
  return routeCost

def population_value(population, costMatrix):
  return [route_cost(chromosome, costMatrix) for chromosome in population]

def get_order(minimization):
  if minimization:
    return 1
  else:
    return -1

def adjust_costMatrix(costMatrix, minimization):
  order = get_order(minimization)
  const = order * np.inf
  for i in range(len(costMatrix)):
      costMatrix[i][i] = const
  return costMatrix

def generate_initial_population(costMatrix, popuSize, minimization, method='Random'):
  initPop = []
  if method == 'Random':
    for n in range(popuSize):
      soln = []
      cities = list(range(len(costMatrix)))
      for i in range(len(cities)):
        randomCity = np.random.choice(cities)
        soln.append(randomCity)
        cities.remove(randomCity)
      initPop.append(soln)
  return initPop

def recombination(sortedParents, crossoverRate=0.95):
  pairs = [(sortedParents[i], sortedParents[j]) for i in range(len(sortedParents)) for j in range(len(sortedParents)) if i!=j]
  children = []
  for pair in pairs:
    startIdx = np.random.randint(len(pair[0])-1)
    endIdx = np.random.randint(startIdx+1, len(pair[0]))
    if (endIdx-startIdx) >= 3:
      endIdx -= 1
    child = pair[0][startIdx:endIdx+1]
    missing = [i for i in pair[1] if i not in child]
    child.extend(missing)

    if np.random.uniform(0,1) <= crossoverRate:
      children.append(child)
  return children

def mutation(unchosenParents, mutationRate=0.1):
  mutated = []
  for parent in unchosenParents:
    choices = list(range(len(parent)))
    randomIdx = np.random.choice(choices)
    choices.remove(randomIdx)
    randomIdx2 = np.random.choice(choices)
    parent[randomIdx2], parent[randomIdx2] = parent[randomIdx], parent[randomIdx2]
    if np.random.uniform(0,1) <= mutationRate:
      mutated.append(parent)
  return mutated

def generate_children(population, crossoverRate, mutationRate, costMatrix, minimization):
  numParents = np.random.randint(2, len(population))
  popuCost = population_value(population, costMatrix)

  sortedParents = [population[i] for i in np.argsort(popuCost)[::get_order(minimization)][:numParents]]  # select top parents from population acc to cost
  unchosenParents = [population[i] for i in np.argsort(popuCost)[::get_order(minimization)][numParents:]]

  children = recombination(sortedParents, crossoverRate)
  mutated = mutation(unchosenParents, mutationRate)
  children.extend(mutated)

  return children
      
def discard_from_population(population, popuSize, costMatrix, iter, minimization):
  newPop = copy.deepcopy(population)
  popuCost = population_value(newPop, costMatrix)
  sortedPopu = [newPop[i] for i in np.argsort(popuCost)[::get_order(minimization)]][:popuSize]
  return sortedPopu

def optimal_soln(solution, nodeLabels, startingNode):
    optSoln = []
    ind = np.where(np.array(solution) == startingNode)[0][0]
    for i in range(ind, len(solution)):
        optSoln.append(i)
    for i in range(ind):
        optSoln.append(i)
    optimal_ind = [solution[i] for i in optSoln]
    optimal_ind.append(optimal_ind[0])
    return [nodeLabels[k] for k in optimal_ind]

def genetic_algorithm_TSP(costMatrix, popuSize, crossoverRate, mutationRate, maxIterations, nodeLabels, minimization=True):
  costMatrix = adjust_costMatrix(costMatrix, minimization)

  initPopu = generate_initial_population(costMatrix, popuSize, minimization)
  print(f'Initial Population: \n Values: {population_value(initPopu, costMatrix)} \n')

  population = initPopu
  worstCosts = []
  popuCosts = []
  for iter in range(1, maxIterations+1):
    print(f'Iteration {iter} ------------------ \n')

    children = generate_children(population, crossoverRate, mutationRate, costMatrix, minimization)
    population.extend(children)
    print(f'Generated Children: \n Values: {sorted(population_value(children, costMatrix))} \n')
    population = discard_from_population(population, popuSize, costMatrix, iter, minimization)
    popuCost = sorted(population_value(population, costMatrix))
    print(f'New Population: \n Values: {popuCost} \n')
    worstCosts.append(popuCost[-1])
    popuCosts.append(sorted(popuCost, reverse=True))
  
  # get optimal solution
  optSoln = population[np.argsort(popuCost)[::get_order(minimization)][0]]
  print('--------------------------------------------')
  print(f'After {iter} iteration(s), optimal solution is {optSoln} \n Route Cost = {route_cost(optSoln, costMatrix)} \n')

  print(f'That is, optimal solution is the path \n {optimal_soln(optSoln, nodeLabels, startingNode=0)}')
  return worstCosts, popuCosts

# Main Implementation of GA

# Parameters
costMatrix = [
    [0, 19, 14, 11, 23, 24],
    [24, 0, 12, 30, 30, 19],
    [40, 42, 0, 20, 36, 15],
    [20, 35, 37, 0, 45, 33],
    [15, 26, 18, 25, 0, 30],
    [22, 17, 14, 30, 28, 0]
]
popuSize = 10
nodeLabels = ["A", "B", "C", "D", "E", "F"]
minimization = True
maxIterations = 20
crossoverRate = 0.70
mutationRate = 0.30

worstCosts, popuCosts = genetic_algorithm_TSP(costMatrix, popuSize, crossoverRate, mutationRate, maxIterations, nodeLabels, minimization)
