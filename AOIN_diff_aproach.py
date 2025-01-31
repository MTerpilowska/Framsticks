import argparse
import os
import sys
import numpy as np
from deap import creator, base, tools, algorithms
from FramsticksLib import FramsticksLib

# Define constants
FITNESS_VALUE_INFEASIBLE_SOLUTION = -999999.0
OPTIMIZATION_CRITERIA = ['vertpos', 'velocity', 'distance', 'vertvel', 'lifespan']  # Example criteria

# Evaluate a creature
def frams_evaluate(frams_lib, individual):
    genotype = individual[0]
    data = frams_lib.evaluate([genotype])
    
    try:
        first_genotype_data = data[0]
        evaluation_data = first_genotype_data["evaluations"]
        default_evaluation_data = evaluation_data[""]
        fitness = [default_evaluation_data[crit] for crit in OPTIMIZATION_CRITERIA]
    except (KeyError, TypeError) as e:
        fitness = [FITNESS_VALUE_INFEASIBLE_SOLUTION] * len(OPTIMIZATION_CRITERIA)
    
    return fitness

# Crossover for genotypes
def frams_crossover(frams_lib, individual1, individual2):
    geno1 = individual1[0]
    geno2 = individual2[0]
    individual1[0] = frams_lib.crossOver(geno1, geno2)
    individual2[0] = frams_lib.crossOver(geno1, geno2)
    return individual1, individual2

# Mutation for genotypes
def frams_mutate(frams_lib, individual):
    individual[0] = frams_lib.mutate([individual[0]])[0]
    return individual,

# Initialize the simplest genotype
def frams_getsimplest(frams_lib, genetic_format, initial_genotype):
    return initial_genotype if initial_genotype is not None else frams_lib.getSimplest(genetic_format)

# Setup DEAP Toolbox
def prepare_toolbox(frams_lib, genetic_format, initial_genotype, tournament_size):
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * len(OPTIMIZATION_CRITERIA))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_simplest_genotype", frams_getsimplest, frams_lib, genetic_format, initial_genotype)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_simplest_genotype, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", frams_evaluate, frams_lib)
    toolbox.register("mate", frams_crossover, frams_lib)
    toolbox.register("mutate", frams_mutate, frams_lib)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    
    return toolbox

# Parse the arguments from the command line
def parseArguments():
    parser = argparse.ArgumentParser(description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[0])
    parser.add_argument('-path', type=ensureDir, required=True, help='Path to Framsticks library without trailing slash.')
    parser.add_argument('-lib', required=False, help='Library name. If not given, "frams-objects.dll" (or .so or .dylib) is assumed depending on the platform.')
    parser.add_argument('-sim', required=False, default="eval-allcriteria.sim", help="The name of the .sim file with settings for evaluation, mutation, crossover, and similarity estimation.")
    parser.add_argument('-genformat', required=False, help='Genetic format for the simplest initial genotype, e.g., 4, 9, or B.')
    parser.add_argument('-initialgenotype', required=False, help='The genotype used to seed the initial population.')
    parser.add_argument('-opt', required=True, help='Optimization criteria: vertpos, velocity, distance, vertvel, lifespan, numjoints, numparts, numneurons, numconnections (or other).')
    parser.add_argument('-popsize', type=int, default=50, help="Population size, default: 50.")
    parser.add_argument('-generations', type=int, default=5, help="Number of generations, default: 5.")
    parser.add_argument('-tournament', type=int, default=5, help="Tournament size, default: 5.")
    parser.add_argument('-pmut', type=float, default=0.9, help="Probability of mutation, default: 0.9")
    parser.add_argument('-pxov', type=float, default=0.2, help="Probability of crossover, default: 0.2")
    parser.add_argument('-hof_size', type=int, default=10, help="Number of genotypes in Hall of Fame. Default: 10.")
    parser.add_argument('-hof_savefile', required=False, help='If set, Hall of Fame will be saved in Framsticks file format.')
    parser.add_argument('-max_numparts', type=int, default=None, help="Maximum number of parts. Default: no limit")
    parser.add_argument('-max_numjoints', type=int, default=None, help="Maximum number of joints. Default: no limit")
    parser.add_argument('-max_numneurons', type=int, default=None, help="Maximum number of neurons. Default: no limit")
    parser.add_argument('-max_numconnections', type=int, default=None, help="Maximum number of connections. Default: no limit")
    parser.add_argument('-max_numgenochars', type=int, default=None, help="Maximum number of characters in genotype. Default: no limit")
    
    return parser.parse_args()

# Ensure that the directory is valid
def ensureDir(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

# Save genotypes to a file
def save_genotypes(filename, OPTIMIZATION_CRITERIA, hof):
    from framsfiles import writer as framswriter
    with open(filename, "w") as outfile:
        for ind in hof:
            keyval = {}
            for i, k in enumerate(OPTIMIZATION_CRITERIA):
                keyval[k] = ind.fitness.values[i]
            outfile.write(framswriter.from_collection({"_classname": "org", "genotype": ind[0], **keyval}))
            outfile.write("\n")
    print(f"Saved '{filename}' ({len(hof)})")

# Main evolution loop
def main():
    parsed_args = parseArguments()

    # Initialize Framsticks library and toolbox
    frams_lib = FramsticksLib(parsed_args.path, parsed_args.lib, parsed_args.sim)
    toolbox = prepare_toolbox(frams_lib, '1', parsed_args.initialgenotype, parsed_args.tournament)
    
    # Create the population
    pop = toolbox.population(n=parsed_args.popsize)
    hof = tools.HallOfFame(parsed_args.hof_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("stddev", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Evolve the population
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=parsed_args.pxov, mutpb=parsed_args.pmut, ngen=parsed_args.generations, 
                                   stats=stats, halloffame=hof, verbose=True)
    
    # Display the best individuals
    print('Best individuals:')
    for ind in hof:
        print(ind.fitness, '\t<--\t', ind[0])

    if parsed_args.hof_savefile is not None:
        save_genotypes(parsed_args.hof_savefile, OPTIMIZATION_CRITERIA, hof)

if __name__ == "__main__":
    main()
