import random

# Define the fitness function (change this to suit your problem)
def fitness_function(solution):
    return sum(solution)

# Define the genetic algorithm
def genetic_algorithm(population_size, chromosome_size, generations):
    # Initialize the population with random solutions
    population = [[random.randint(0, 1) for _ in range(chromosome_size)] for _ in range(population_size)]
    
    for generation in range(generations):
        # Evaluate the fitness of each solution
        fitness_scores = [fitness_function(solution) for solution in population]
        
        # Select the best solutions for the next generation
        selected_population = []
        for _ in range(population_size):
            parent1 = population[random.randint(0, population_size-1)]
            parent2 = population[random.randint(0, population_size-1)]
            selected_population.append(max([parent1, parent2], key=fitness_function))
        
        # Generate new solutions through crossover and mutation
        new_population = []
        for i in range(population_size):
            parent1 = selected_population[random.randint(0, population_size-1)]
            parent2 = selected_population[random.randint(0, population_size-1)]
            crossover_point = random.randint(0, chromosome_size-1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            for j in range(chromosome_size):
                if random.random() < 0.1: # Mutation rate of 10%
                    child[j] = 1 - child[j]
            new_population.append(child)
        
        # Update the population
        population = new_population
    
    # Return the best solution
    return max(population, key=fitness_function)

#To use the genetic algorithm, simply call the genetic_algorithm() function with the desired population size, chromosome size, and number of generations
solution = genetic_algorithm(population_size=100, chromosome_size=10, generations=100)
print(solution)

