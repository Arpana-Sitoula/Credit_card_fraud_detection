import random
import numpy as np
import pandas as pd
#from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix

# Load the credit card dataset
df = pd.read_csv(r'C:\Users\abi3c\Desktop\creditcard.csv')
data = df.drop('Class' ,axis = 1)
target = df['Class']
X = data.to_numpy()
y = target.to_numpy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the fitness function
def fitness_function(individual):
    # Create a logistic regression model with the selected features
    model = LogisticRegression(solver='liblinear', C=1, max_iter=100)
    X_train_subset = X_train[:, individual]
    X_test_subset = X_test[:, individual]
    # Train the model and make predictions on the test set
    model.fit(X_train_subset, y_train)
    y_pred = model.predict(X_test_subset)
    # Compute the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)
    # Return the accuracy as the fitness score
    return accuracy

# Define the genetic operators
def selection(population, fitness_fn, num_parents):
    # Select the fittest individuals as parents
    parents = []
    for i in range(num_parents):
        parent = max(population, key=fitness_fn)
        parents.append(parent)
        population.remove(parent)
    return parents

def crossover(parents, offspring_size):
    offspring = []
    for i in range(offspring_size):
        # Choose two parents randomly
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        # Choose a random crossover point
        crossover_point = random.randint(0, len(parent1))
        # Create the offspring by combining the parents' genes
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        offspring.append(offspring1)
        offspring.append(offspring2)
    return offspring

def mutation(population, mutation_prob):
    # Mutate each individual in the population with the specified probability
    for i in range(len(population)):
        for j in range(len(population[i])):
            if random.random() < mutation_prob:
                population[i][j] = not population[i][j]

# Set the genetic algorithm parameters
POP_SIZE = 50
NUM_PARENTS = 10
NUM_GENERATIONS = 50
MUTATION_PROB = 0.1

# Initialize the population with random individuals
population = []
for i in range(POP_SIZE):
    individual = [random.choice([True, False]) for j in range(X_train.shape[1])]
    population.append(individual)

# Run the genetic algorithm
for generation in range(NUM_GENERATIONS):
    # Select the parents for the next generation
    parents = selection(population, fitness_function, NUM_PARENTS)
    # Generate the offspring for the next generation
    offspring = crossover(parents, POP_SIZE - NUM_PARENTS)
    # Mutate the offspring
    mutation(offspring, MUTATION_PROB)
    # Combine the parents and offspring to form the next generation
    population = parents + offspring

# Select the best individual from the final population
best_individual = max(population, key=fitness_function)
best_fitness = fitness_function(best_individual)

print("Selected features:", best_individual)
print("Accuracy:", best_fitness)



# Initialize logistic regression model
lr_model = LogisticRegression()

# Fit the model on the selected features
lr_model.fit(X_train[:, best_individual], y_train)

# Make predictions on the test set
y_pred = lr_model.predict(X_test[:, best_individual])

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", conf_matrix)

# Generate classification report
class_report = classification_report(y_test, y_pred)
print("Classification report:\n", class_report)
