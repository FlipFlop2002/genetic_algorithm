from functions import *

pc = 0.7  # crossover probability
pm = 0.5  # mutation probability
population_size = 20
tmax = 40

results = []
for i in range(10):


    population = generate_population(population_size)
    rating = evaluate(population)
    final_best_solution = find_best(population)
    # plot_graph(final_best_solution, "07_07_20_40", save=False)

    t = 1
    while t <= tmax:
        # temp_population = proportionate_selection(population, rating)
        temp_population = tournament_selection(population)
        new_population = crossover_mutation(temp_population, pc, pm)
        rating = evaluate(new_population)
        best_solution = find_best(new_population)

        if best_solution.distance < final_best_solution.distance:
            final_best_solution = best_solution


        population = new_population
        t += 1

    results.append(final_best_solution)



dist_sum = 0
for solution in results:
    print(round(solution.distance, 4), solution.chromosome)
    dist_sum += solution.distance

print(f'distance sum: {dist_sum}')
print(f'Avarage distance: {dist_sum / len(results)}')




plot_graph(find_best(results), "tournament_07_10_20_40", save=True)