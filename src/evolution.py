from deap import tools, algorithms
import search
from copy import deepcopy

# This is a modified version of the eaSimple algorithm from the DEAP library (https://github.com/DEAP/deap/blob/master/deap/algorithms.py). Modified to include gradient descent
def gradientEvolution(population, toolbox, cxpb, mutpb, ngen, xs, ys, context, arguments, classes, gd_frequency, epochs, lr,  extended=False, patience=10, stats=None,
             halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    no_improvement = 0 # Number of iterations without improvement in max
    max_fitness = 0
    original_mutpb = mutpb

    # Run on entire population
    num_best = len(population) #// 10

    # Begin the generational process
    for gen in range(0, ngen):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        last_max = max_fitness

        for ind, fit in zip(invalid_ind, fitnesses):
            max_fitness = max(max_fitness, fit[0])
            ind.fitness.values = fit

        # Track how many gens without improvement
        if max_fitness == last_max:
            no_improvement += 1
        else:
            no_improvement = 0
            mutpb = original_mutpb# Reset mutation rate

        # If we havent been progressing, keep increasing the mutation
        if no_improvement >= patience:
            increase = original_mutpb * 0.1 # 10% Increase
            cxpb -= increase # Decrease crossover
            if verbose:
                print("Increasing mutation due to no improvement", mutpb, increase)
            mutpb += increase # Increase mutation

        # On final generation apply gradient descent on fittest individual only
        if gd_frequency != -1 and (gen == ngen or max_fitness == 1):
            if verbose:
                print("Applying gradient descent on fittest individual")

            fittest_index = 0
            highest_fitness = 0

            # Find fittest individual
            for idx, individual in enumerate(offspring):
                fitness = ind.fitness.values[0]

                if fitness > highest_fitness:
                    highest_fitness = fitness
                    fittest_index = idx

            fittest_ind = offspring[fittest_index]

            if extended:
                epochs = 100
                
            updated_fittest_ind = search.gradient_descent(fittest_ind, xs, ys, context, arguments, classes, epochs, lr)

            # Update the individual on offspring
            offspring[fittest_index] = updated_fittest_ind

        # Apply gradient descent every n generations
        elif gd_frequency != -1 and gen % gd_frequency == 0:
            if verbose:
                print("Applying gradient descent", gen)

            # Sort the offspring in descending fitness
            best_individuals = sorted(offspring, reverse=True, key=lambda ind: ind.fitness.values[0])

            # Run gradient descent on the best, leaving the rest unchanged
            cache = {} #Save the tree ->updated tree map, so in the case of duplicated trees in the best individuals we do not need to run gradient descent again
            updated_best = []

            for ind in best_individuals[:num_best]:
                tree_str = str(ind)

                # If we have already seen this tree, use the original value (dont run gradient descent!)
                if tree_str in cache:
                    updated = deepcopy(cache[tree_str])
                # Otherwise need to run gradient descent and store in cashe
                else:
                    updated = search.gradient_descent(ind, xs, ys, context, arguments, classes, epochs, lr)
                    cache[tree_str] = updated

                updated_best.append(updated)

            best_individuals[:num_best] = updated_best

            # Update the offspring
            offspring = best_individuals
            

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)


        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

         # Early exit, we achieved top fitness
        if max_fitness == 1:
            break


    return population, logbook