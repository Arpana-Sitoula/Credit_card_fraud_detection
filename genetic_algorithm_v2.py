###############################################################################
#                         Genetic Algorithm                                  #
###############################################################################

# ----------------------- Importing libraries -----------------------
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import random
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# Initialization

def initilialize_poplulation(numberOfParents):
    '''
    Initialization: 
    The parameters are randomly initialized to create the population. 
    It is the first generation of the population. 
    We generate a vector containing the hyperparameters. 
    We have selected 7 hyperparameters to optimze: 
    - learning rate
    - n_estimators
    - min_child_wight 
    - subsample
    - colsample_bytree
    - gamma
    '''
    learningRate = np.empty([numberOfParents, 1])
    nEstimators = np.empty([numberOfParents, 1])
    maxDepth = np.empty([numberOfParents, 1])
    minChildWeight = np.empty([numberOfParents, 1])
    gammaValue = np.empty([numberOfParents, 1])
    subSample = np.empty([numberOfParents, 1])
    colSampleByTree =  np.empty([numberOfParents, 1])
    for i in range(numberOfParents):
        print(i)
        learningRate[i] = round(random.uniform(0.01, 1), 2)
        nEstimators[i] = random.randrange(10, 150, step = 25)
        maxDepth[i] = int(random.randrange(1, 10, step= 1))
        minChildWeight[i] = round(random.uniform(0.01, 10.0), 2)
        gammaValue[i] = round(random.uniform(0.01, 10.0), 2)
        subSample[i] = round(random.uniform(0.01, 1.0), 2)
        colSampleByTree[i] = round(random.uniform(0.01, 1.0), 2)
    population = np.concatenate((learningRate, nEstimators, maxDepth, minChildWeight, gammaValue, subSample, colSampleByTree), axis= 1)
    return population


def fitness_function(y_true, y_pred):
    '''
    Fitness Function:
      In this case we will calculate the F-1 score as fitness function. 
      You can chose Accuracy as the fitness function or whatever metrics that 
      you would like to minimize
    '''
    fitness = round((f1_score(y_true, y_pred, average='weighted')), 4)
    return fitness


#train the data annd find fitness score
def train_population(population, X_train, y_train, X_test, y_test):
    firness_score = []
    for i in range(population.shape[0]):
        param = { 'objective':'binary:logistic',
                'learning_rate': population[i][0],
                'n_estimators': int(population[i][1]), 
                'max_depth': int(population[i][2]), 
                'min_child_weight': population[i][3],
                'gamma': population[i][4], 
                'subsample': population[i][5],
                'colsample_bytree': population[i][6],
                'seed': 24}
        XGBC = XGBClassifier(n_jobs = 4,random_state = 123).set_params(**param)
        model = XGBC.fit(X_train, y_train)
        preds = model.predict(X_test)
        # preds = preds>0.5
        firness_score.append(fitness_function(y_test, preds))
    return firness_score



#select parents for mating
def new_parents_selection(population, fitness, numParents):
    selectedParents = np.empty((numParents, population.shape[1])) #create an array to store fittest parents
    #find the top best performing parents
    for parentId in range(numParents):
        bestFitnessId = np.where(fitness == np.max(fitness))
        bestFitnessId  = bestFitnessId[0][0]
        selectedParents[parentId, :] = population[bestFitnessId, :]
        fitness[bestFitnessId] = -1 #set this value to negative, in case of F1-score, so this parent is not selected again
    return selectedParents


# crossover function
def crossover_uniform(parents, childrenSize):
    '''
    Crossover
    Mate these parents to create children having parameters from these parents 
    (we are using uniform crossover method)
    There are various methods to define crossover in the case of genetic algorithms, 
    such as single-point, two-point and k-point crossover, uniform crossover and crossover 
    for ordered lists. We are going to use uniform crossover, where each parameter 
    for the child will be independently selected from the parents, based on a certain 
    distribution. In our case we will use “discrete uniform” distribution from numpy random function .
    
    '''
    crossoverPointIndex = np.arange(0, np.uint8(childrenSize[1]), 1, dtype= np.uint8) #get all the index
    crossoverPointIndex1 = np.random.randint(0, np.uint8(childrenSize[1]), np.uint8(childrenSize[1]/2)) # select half  of the indexes randomly
    crossoverPointIndex2 = np.array(list(set(crossoverPointIndex) - set(crossoverPointIndex1))) #select leftover indexes
    children = np.empty(childrenSize)
    '''
    Create child by choosing parameters from two parents selected using new_parent_selection function. The parameter values
    will be picked from the indexes, which were randomly selected above. 
    '''
    for i in range(childrenSize[0]):
        
        #find parent 1 index 
        parent1_index = i%parents.shape[0]
        #find parent 2 index
        parent2_index = (i+1)%parents.shape[0]
        #insert parameters based on random selected indexes in parent 1
        children[i, crossoverPointIndex1] = parents[parent1_index, crossoverPointIndex1]
        #insert parameters based on random selected indexes in parent 1
        children[i, crossoverPointIndex2] = parents[parent2_index, crossoverPointIndex2]
    return children


# Mutation
def mutation(crossover, numberOfParameters):
    #Define minimum and maximum values allowed for each parameter
    minMaxValue = np.zeros((numberOfParameters, 2))
    minMaxValue[0:] = [0.01, 1.0] #min/max learning rate
    minMaxValue[1, :] = [10, 2000] #min/max n_estimator
    minMaxValue[2, :] = [1, 15] #min/max depth
    minMaxValue[3, :] = [0, 10.0] #min/max child_weight
    minMaxValue[4, :] = [0.01, 10.0] #min/max gamma
    minMaxValue[5, :] = [0.01, 1.0] #min/maxsubsample
    minMaxValue[6, :] = [0.01, 1.0] #min/maxcolsample_bytree
    # Mutation changes a single gene in each offspring randomly.
    mutationValue = 0
    parameterSelect = np.random.randint(0, 7, 1)
    print(parameterSelect)
    if parameterSelect == 0: #learning_rate
        mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
    if parameterSelect == 1: #n_estimators
        mutationValue = np.random.randint(-200, 200, 1)
    if parameterSelect == 2: #max_depth
        mutationValue = np.random.randint(-5, 5, 1)
    if parameterSelect == 3: #min_child_weight
        mutationValue = round(np.random.uniform(5, 5), 2)
    if parameterSelect == 4: #gamma
        mutationValue = round(np.random.uniform(-2, 2), 2)
    if parameterSelect == 5: #subsample
        mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
    if parameterSelect == 6: #colsample
        mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
    #indtroduce mutation by changing one parameter, and set to max or min if it goes out of range
    for idx in range(crossover.shape[0]):
        crossover[idx, parameterSelect] = crossover[idx, parameterSelect] + mutationValue
        if(crossover[idx, parameterSelect] > minMaxValue[parameterSelect, 1]):
            crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 1]
        if(crossover[idx, parameterSelect] < minMaxValue[parameterSelect, 0]):
            crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 0]    
    return crossover


# Implementing the GA 

# Dataset
# X, y = make_classification(
#     n_samples=100000, # 1000 observations 
#     n_features=150, # 5 total features
#     n_informative=3, # 3 'useful' features
#     n_classes=2, # binary target/label 
#     random_state=999 # if you want the same results as mine
# )
DataFrame = pd.read_csv(r'C:/Users/abi3c/Desktop/creditcard.csv')

X = DataFrame.drop(columns="Class", axis=1)
y = DataFrame["Class"]

#X = pd.DataFrame(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


numberOfParents = 8 #number of parents to start
numberOfParentsMating = 4 #number of parents that will mate
numberOfParameters = 7 #number of parameters that will be optimized
numberOfGenerations = 4 #number of genration that will be created

#define the population size
populationSize = (numberOfParents, numberOfParameters)

#initialize the population with randomly generated parameters
population = initilialize_poplulation(numberOfParents)

#define an array to store the fitness  hitory
fitnessHistory = np.empty([numberOfGenerations+1, numberOfParents])

#define an array to store the value of each parameter for each parent and generation
populationHistory = np.empty([(numberOfGenerations+1)*numberOfParents, numberOfParameters])
#insert the value of initial parameters in history
populationHistory[0:numberOfParents, :] = population

for generation in range(numberOfGenerations):
    print("This is number %s generation" % (generation))
    #train the dataset and obtain fitness
    fitnessValue = train_population(population, X_train, y_train, X_test, y_test)
    fitnessHistory[generation, :] = fitnessValue
    #best score in the current iteration
    print('Best F1 score in the this iteration = {}'.format(np.max(fitnessHistory[generation, :])))
    #survival of the fittest - take the top parents, based on the fitness value and number of parents needed to be selected
    parents = new_parents_selection(population=population, fitness=fitnessValue, numParents=numberOfParentsMating)
    #mate these parents to create children having parameters from these parents (we are using uniform crossover)
    children = crossover_uniform(parents=parents, childrenSize=(populationSize[0] - parents.shape[0], numberOfParameters))
    #add mutation to create genetic diversity
    children_mutated = mutation(children, numberOfParameters)
    '''
    We will create new population, which will contain parents that where selected previously based on the
    fitness score and rest of them  will be children
    '''
    population[0:parents.shape[0], :] = parents #fittest parents
    population[parents.shape[0]:, :] = children_mutated #children
    populationHistory[(generation+1)*numberOfParents : (generation+1)*numberOfParents+ numberOfParents , :] = population #srore parent information


#Best solution from the final iteration
fitness = train_population(population, X_train, y_train, X_test, y_test)
fitnessHistory[generation+1, :] = fitness
#index of the best solution
bestFitnessIndex = np.where(fitness == np.max(fitness))[0][0]
#Best fitness
print("Best fitness is =", fitness[bestFitnessIndex])
#Best parameters
print("Best parameters are:")
print('learning_rate', population[bestFitnessIndex][0])
print('n_estimators', population[bestFitnessIndex][1])
print('max_depth', int(population[bestFitnessIndex][2])) 
print('min_child_weight', population[bestFitnessIndex][3])
print('gamma', population[bestFitnessIndex][4])
print('subsample', population[bestFitnessIndex][5])
print('colsample_bytree', population[bestFitnessIndex][6])
