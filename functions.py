import numpy as np
from math import sqrt
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
import time

class Solution:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.distance = None



def calculate_single_distance(city1: list, city2: list):
    d = sqrt((city2[0]-city1[0])**2 + (city2[1]-city1[1])**2)
    return d


def calculate_travel_distance(chromosome: np.ndarray):
    coordinates = {0: [0, 0], 1: [1, 3], 2: [2, 1], 3: [4, 6], 4: [5, 2], 5: [6, 5], 6: [8, 7], 7: [9, 4], 8: [10, 8], 9: [12, 3]}
    distance = 0
    for i in range(chromosome.shape[0]):
        if i == chromosome.shape[0] - 1:
            d = calculate_single_distance(coordinates[chromosome[i]], coordinates[chromosome[0]])
            distance += d
        else:
            d = calculate_single_distance(coordinates[chromosome[i]], coordinates[chromosome[i+1]])
            distance += d

    return distance


def generate_population(population_size: int) -> list:
    population = []
    for i in range(population_size):
        chromosome = np.arange(10)
        np.random.shuffle(chromosome)
        s = Solution(chromosome)
        s.distance = calculate_travel_distance(s.chromosome)
        population.append(s)

    return population


def evaluate(population: list):
    distances_list = []
    for solution in population:
        distances_list.append(solution.distance)

    reversed_ = [1 / x for x in distances_list]
    sum_ = sum(reversed_)
    reversed_ = [x / sum_ for x in reversed_]
    return reversed_


def proportionate_selection(population, eval_scores):
    k = len(population)
    return random.choices(population, eval_scores, k=k)


def tournament_selection(population):
    selected = []
    for i in range(len(population)):
        pair = random.choices(population, k=2)
        if pair[0].distance == pair[1].distance:
            selected.append(pair[0])
        elif pair[0].distance > pair[1].distance:
            selected.append(pair[1])
        else:
            selected.append(pair[0])

    return selected


def crossover_mutation(temporary_population: list, pc, pm):
    new_population = []
    mi = len(temporary_population)
    # creating pairs
    pairs = np.arange(mi)
    np.random.shuffle(pairs)

    # crossover
    for i in range(mi - 1):
        if i % 2 == 0:
            # crossover probability
            if random.choices([0, 1], [1 - pc, pc]):

                # choose crossover point
                co_idx = random.randint(1, len(temporary_population[0].chromosome) - 1)

                parent1 = temporary_population[pairs[i]]
                parent2 = temporary_population[pairs[i+1]]

                child1 = parent1.chromosome[:co_idx]

                for gen in parent2.chromosome:
                    if gen in parent1.chromosome[co_idx:]:
                        child1 = np.append(child1, gen)
                s1 = Solution(child1)


                child2 = parent2.chromosome[:co_idx]
                for gen in parent1.chromosome:
                    if gen in parent2.chromosome[co_idx:]:
                        child2 = np.append(child2, gen)
                s2 = Solution(child2)

                new_population.append(s1)
                new_population.append(s2)

            else:
                new_population.append(temporary_population[pairs[i]])
                new_population.append(temporary_population[pairs[i+1]])


    # mutation
    size = 10
    for solution in new_population:
        if random.choices([0, 1], [1 - pm, pm]):
            idx1 = random.randint(0, size-1)
            idx2 = random.randint(0, size-1)
            while(idx1 == idx2):
                idx2 = random.randint(0, size-1)


            tmp = solution.chromosome[idx1]
            solution.chromosome[idx1] = solution.chromosome[idx2]
            solution.chromosome[idx2] = tmp


        solution.distance = calculate_travel_distance(solution.chromosome)

    return new_population


def find_best(population):
    best_solution = population[0]
    for solution in population:
        if best_solution.distance > solution.distance:
            best_solution = solution
    return best_solution


def plot_graph(solution: Solution, name: str, save: bool):
    coordinates = {0: [0, 0], 1: [1, 3], 2: [2, 1], 3: [4, 6], 4: [5, 2], 5: [6, 5], 6: [8, 7], 7: [9, 4], 8: [10, 8], 9: [12, 3]}
    cities = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    x = []
    y = []
    for point in list(coordinates.values()):
        x.append(point[0])
        y.append(point[1])

    x_sol = []
    y_sol = []
    for point in solution.chromosome:
        x_sol.append(coordinates[point][0])
        y_sol.append(coordinates[point][1])

    x_sol.append(coordinates[solution.chromosome[0]][0])
    y_sol.append(coordinates[solution.chromosome[0]][1])

    route = ''
    for city in solution.chromosome:
        route = route + cities[city]

    plt.figure(figsize=(8, 6))
    plt.plot(x_sol, y_sol, 'bo-')
    plt.plot(x, y, 'bo')  # 'bo' oznacza niebieskie kropki (blue circles)
    plt.plot(x_sol[0], y_sol[0], 'ro')
    plt.title(f'Łączny dystans: {solution.distance}, Trasa: {route}')
    plt.xlabel('Oś X')
    plt.ylabel('Oś Y')
    plt.grid(True)
    for i in range(len(x)):
        plt.text(x[i], y[i], cities[i], fontsize=10, ha='center', va='bottom')
    if save:
        plt.savefig(f'graf_{name}.pdf')
    plt.show()




# def plot_graph_live(solution: Solution, name: str):
#     coordinates = {0: [0, 0], 1: [1, 3], 2: [2, 1], 3: [4, 6], 4: [5, 2], 5: [6, 5], 6: [8, 7], 7: [9, 4], 8: [10, 8], 9: [12, 3]}
#     cities = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
#     x = []
#     y = []
#     for point in list(coordinates.values()):
#         x.append(point[0])
#         y.append(point[1])

#     x_sol = []
#     y_sol = []
#     for point in solution.chromosome:
#         x_sol.append(coordinates[point][0])
#         y_sol.append(coordinates[point][1])

#     x_sol.append(coordinates[solution.chromosome[0]][0])
#     y_sol.append(coordinates[solution.chromosome[0]][1])

#     route = ''
#     for city in solution.chromosome:
#         route = route + cities[city]

#     plt.figure(figsize=(8, 6))
#     plt.plot(x_sol, y_sol, 'bo-')
#     plt.plot(x, y, 'bo')  # 'bo' oznacza niebieskie kropki (blue circles)
#     plt.plot(x_sol[0], y_sol[0], 'ro')
#     plt.title(f'Łączny dystans: {solution.distance}, Trasa: {route}')
#     plt.xlabel('Oś X')
#     plt.ylabel('Oś Y')
#     plt.grid(True)
#     for i in range(len(x)):
#         plt.text(x[i], y[i], cities[i], fontsize=10, ha='center', va='bottom')
#     plt.savefig(f'graf_{name}.pdf')
#     plt.show()