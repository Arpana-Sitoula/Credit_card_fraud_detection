###############################################################################
#                         Genetic Algorithm                                  #
###############################################################################

# ----------------------- Importing libraries -----------------------
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# Initialization

def initilialize_poplulation(numberOfParents):
    maxDepth = np.empty([numberOfParents, 1]) 
    nEstimators = np.empty([numberOfParents, 1])
    minSamplesSplit = np.empty([numberOfParents,1])
    minSamplesLeaf = np.empty([numberOfParents, 1])
    maxFeatures = np.empty([numberOfParents, 1])
    bootStrap = np.empty([numberOfParents, 1])
    minImpurityDesc =  np.empty([numberOfParents, 1])
    for i in range(numberOfParents):
        print(i)
        maxDepth[i] = int(random.randrange(1, 10, step= 1))
        nEstimators[i] = random.randrange(10, 150, step = 25)
        minSamplesSplit[i] = round(random.uniform(0.01, 1.0), 2)
        minSamplesLeaf[i] = round(random.uniform(0.01, 10.0), 2)
        maxFeatures[i] = int(random.randrange(1, 10, step= 2))
        bootStrap[i] = bool(random.choice([True, False]))
        minImpurityDesc[i] = round(random.uniform(0.01, 1.0), 2)
    population = np.concatenate((maxDepth,nEstimators, minSamplesSplit, minSamplesLeaf, maxFeatures, bootStrap, minImpurityDesc), axis= 1)
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

## we can also use others evaluation metrics like accuracy, precision or recall 
def accuracy_function(y_true, y_pred):
    accuracy = round((accuracy_score(y_true, y_pred, average='weighted')),4)
    return accuracy

def precision_function(y_true, y_pred):
    fitness = round((f1_score(y_true, y_pred, average='weighted')), 4)
    return fitness


#train the data annd find fitness score
def train_population(population, X_train, y_train, X_test, y_test):
    fitness_score = []
    for i in range(population.shape[0]):
        param = { 
                'criterion': 'gini',
                'max_depth': int(population[i][0]),
                'n_estimators': int(population[i][1]), 
                'min_samples_split': population[i][2], 
                'min_samples_leaf': population[i][3],
                'max_features': int(population[i][4]), 
                'bootstrap': bool(population[i][5]),
                'min_impurity_decrease': population[i][6],
                }
        rfc = RandomForestClassifier(random_state = 123).set_params(**param)
        model = rfc.fit(X_train, y_train)
        preds = model.predict(X_test)
        # preds = preds>0.5
        fitness_score.append(fitness_function(y_test, preds))
    return fitness_score



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
    minMaxValue[0:] = [1, 10] #min/max depth
    minMaxValue[1, :] = [10, 1000] #min/max n_estimator
    minMaxValue[2, :] = [0.01, 1.0] #min/max sample split
    minMaxValue[3, :] = [0.01, 10.0] #min/max sample leaf
    minMaxValue[4, :] = [1, 10] #min/max features
    minMaxValue[5, :] = [False, True] #min/maxsubsample
    minMaxValue[6, :] = [0.01, 1.0] #min/max min impurity desc
    # Mutation changes a single gene in each offspring randomly.
    mutationValue = 0
    parameterSelect = np.random.randint(0, 7, 1)
    print(parameterSelect)
    if parameterSelect == 0: #max depth
        mutationValue = np.random.randint(-5, 5, 1)
    if parameterSelect == 1: #n_estimators
        mutationValue = np.random.randint(-200, 200, 1)
    if parameterSelect == 2: #min sample split
        mutationValue = round(np.random.uniform(5, 5), 2)
    if parameterSelect == 3: #min sample leaf
        mutationValue = round(np.random.uniform(5, 5), 2)
    if parameterSelect == 4: #max features
        mutationValue = np.random.randint(-2, 2 , 2)
    if parameterSelect == 5: #bootstrap
        mutationValue = bool(random.choice([True, False]))
    if parameterSelect == 6: #min impure desc
        mutationValue = round(np.random.uniform(5, 5), 2)
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
#Getting the datasets
DataFrame = pd.read_csv(r'C:/Users/abi3c/Desktop/creditcard.csv')
DataFrame.drop_duplicates(inplace=True)

#balancing the datasets
non_fraud = DataFrame[DataFrame['Class']==0]
fraud = DataFrame[DataFrame['Class']==1]
legit = non_fraud.sample(n=508)
NewDataFrame = pd.concat([legit,fraud], axis = 0)
X = NewDataFrame.drop(columns="Class", axis=1)
y = NewDataFrame["Class"]

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
