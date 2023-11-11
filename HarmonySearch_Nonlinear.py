## Solving NonLinear Optimization Problem using HARMONY SEARCH ALGORITHM

# packages
import numpy as np
import random

# Helper Functions
def func(soln: list):
    x, y, z = soln[0], soln[1], soln[2]
    return x**2 + y**2 + z**2 - 6*x + 4*y - 2*z

def initialHarmonyMemory(memorySize: int, numVars: int, interval: tuple):
    return [random.sample(range(interval[0], interval[1]+1), numVars) for i in range(memorySize)]

def memoryValues(harmonyMemory: list):
    return [func(soln) for soln in harmonyMemory]

def generateNewSoln(harmonyMemory: list, HMCR: float):
    prob = random.uniform(0, 1)
    if prob <= HMCR:  # random selection
        decision = 'Memory Consideration'
        numVars, numChoices = len(harmonyMemory[0]), len(harmonyMemory)
        inds = random.sample(range(0, numChoices), numVars)
        newSoln = [harmonyMemory[i][idx] for idx,i in enumerate(inds)]
    else:
        decision = 'Random Selection'
        newSoln = random.choice(harmonyMemory)

    return newSoln, decision

def pitchAdjustment(newSoln: list, delta: float, PAR: float):
    prob = random.uniform(0,1)
    if prob <= PAR:  # adjust soln
        plus_min = [random.choice([-1,1])*delta for i in range(len(newSoln))]
        return [sum(x) for x in zip(newSoln, plus_min)]
    else:
        return newSoln

def updateHarmonyMemory(harmonyMemory: list, adjustedSoln: list, minimization: bool):
    hmValues = memoryValues(harmonyMemory)
    newValue = func(adjustedSoln)
    
    if minimization:
        worstInd = np.argmax(hmValues)
        order = 1
    else:
        worstInd = np.argmin(hmValues)
        order = -1

    worstSoln = harmonyMemory[worstInd]
    decision = 'retained'
    if order*newValue < order*hmValues[worstInd]:
        decision = 'updated'
        harmonyMemory.pop(worstInd)
        harmonyMemory.insert(worstInd, adjustedSoln)
    
    return harmonyMemory, worstSoln, decision

def optimalSoln(finalHM: list, minimization: bool):
    hmValues = memoryValues(finalHM)

    if minimization:
        bestInd = np.argmin(hmValues)
    else:
        bestInd = np.argmax(hmValues)
    
    return finalHM[bestInd]

# Main Function
def HarmonySearch_NonLinear(numVars:int, HMS:int, interval:tuple, HMCR:float, PAR:float, delta:float, maxIterations:int, minimization:bool):
    # initial Harmony Memory (HM)
    initHM = initialHarmonyMemory(HMS, numVars, interval)
    print(f'Initial Harmony Memory (HM):\n {initHM} \n Values: {memoryValues(initHM)} \n')
    harmonyMemory = initHM

    continueIter = True
    iterCount = 0
    while continueIter:
        print(f'ITERATION {iterCount+1}: -------------- \n')
        # Generate new solution
        newSoln, decision = generateNewSoln(harmonyMemory, HMCR)
        print(f'New Solution ({decision}): \n {newSoln}, Value: {func(newSoln)} \n')

        # adjust new solution (if stated)
        adjustedSoln = pitchAdjustment(newSoln, delta, PAR)
        print(f'Adjusted Solution:\n {adjustedSoln}, Value: {func(adjustedSoln)} \n')
        
        # update HM
        harmonyMemory, worstSoln, decision = updateHarmonyMemory(harmonyMemory, adjustedSoln, minimization)
        print(f'Worst Harmony: \n {worstSoln}, Value: {func(worstSoln)} \n')
        print(f'New HM ({decision}): \n {harmonyMemory} \n Values: {memoryValues(harmonyMemory)} \n')

        iterCount += 1
        if iterCount == maxIterations:
            continueIter = False
    
    optSoln = optimalSoln(harmonyMemory, minimization)
    print(f'After {iterCount} Iterations, optimal solution: \n [x,y,z]={optSoln}, Value: {func(optSoln)}')


# Parameters
numVars = 3
HMS = 5  # harmony memory size
interval = (-10, 10)  # range of values
HMCR = 0.9  # HM consideration rate
PAR = 0.4  # pitch adjustment rate
delta = 1  # small increments
minimization = True
maxIters = 50

# HM Implementation
HarmonySearch_NonLinear(numVars, HMS, interval, HMCR, PAR, delta, maxIters, minimization)