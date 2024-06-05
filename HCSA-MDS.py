import networkx as nx
import numpy as np
import random
from math import gamma
import math
import glob
import time
from operator import itemgetter
import sys
import os

script_dir = os.path.dirname(__file__)

# Helper functions
def RandomPopulation(G):
    Graflength = len(G.nodes())
    ReducedPopulation = []
    for i in range(40):
        solution = np.random.choice([0, 1], size=(Graflength,), p=[.5 / 4, 3.5 / 4])
        solution = solution.tolist()
        ReducedPopulation.append(solution)
    return ReducedPopulation


def fitness(G, sol):
    V = len(G.nodes())
    vc = set()
    numberOfNodes = sum(sol)
    for index, value in enumerate(sol):
        if value == 1:
            vc.add(index)
            vc.update(G.neighbors(index))
    if numberOfNodes > 0:
        f = (len(vc) / V) + (1 / (V * numberOfNodes))
    else:
        numberOfNodes = len(G.nodes())
        f = (len(vc) / V) + (1 / (V * numberOfNodes))
    return f


def filtering(G, sol):
    best_fitness = fitness(G, sol)
    sol_var = sol.copy()
    for index, value in enumerate(sol):
        if value == 1:
            sol_var[index] = 0
            fit_of_sol_var = fitness(G, sol_var)
            if fit_of_sol_var > best_fitness:
                best_fitness = fit_of_sol_var
                sol[index] = 0
            else:
                sol_var[index] = 1
    return best_fitness, sol


def repair(G, sol):
    vc = {index for index, value in enumerate(sol) if value == 1}
    flat_vc = vc.union(*(set(G.neighbors(node)) for node in vc))
    non_covered = set(G.nodes()) - flat_vc

    while non_covered:
        H = G.subgraph(non_covered)
        degrees = H.degree(non_covered)
        x = max(degrees, key=itemgetter(1))[0]
        non_covered.remove(x)
        sol[x] = 1
        flat_vc.update(G.neighbors(x))
        non_covered.difference_update(G.neighbors(x))

    return sol


def levy_flight(G, worst, beta, sigma, sumxBest):
    for _ in range(1000):
        # Calculate u and v
        u = np.random.normal(0, sigma**2)
        v = np.random.normal(0, sigma**2)
        # Step size
        s = sigma * u / (abs(v) ** (1 / beta))
        # Calculate levy flight
        sumworst = sum(worst)
        Xnew = s * (sumworst - sumxBest)
        if 0 < Xnew <= 1:
            break

    # # Calculate u and v
    # u = np.random.normal(0, sigma ** 2)
    # v = np.random.normal(0, sigma ** 2)
    # # Step size
    # s = sigma * u / (abs(v) ** (1 / beta))
    # # Calculate levy flight
    # sumworst = sum(worst)
    # Xnew = s * (sumworst - sumxBest)

    # Clip Xnew to avoid overflow in the sigmoid function
    Xnew = np.clip(Xnew, -500, 500)
    # Transform Xnew to be within the range [0, 1] using the sigmoid function
    Xnew = 1 / (1 + np.exp(-Xnew))

    n = len(G.nodes())
    h = 3
    k = 4
    Lmax = round(n / h) + 1
    stp = round(Lmax / k)
    step = []

    if 0 <= Xnew < 0.25:
        step = list(range(1, stp + 1))
    elif 0.25 <= Xnew < 0.5:
        step = list(range(stp, 2 * stp))
    elif 0.5 <= Xnew < 0.75:
        step = list(range(2 * stp, 3 * stp))
    elif 0.75 <= Xnew <= 1:
        step = list(range(3 * stp, Lmax + 1))

    # Choose a random number
    Rnumber = np.random.choice(step)
    # Choose a random position
    start = random.randint(0, len(worst) - Rnumber)

    for i in range(start, start + Rnumber):
        worst[i] = 1 - worst[i]

    return worst


def crossover(G, x, y):
    h = random.randint(2, len(x) - 1)
    nocol = len(x)

    child1 = x[:h] + y[h:]
    child2 = y[:h] + x[h:]

    repaired1 = repair(G, child1)
    best_fit1, sol1 = filtering(G, repaired1)
    repaired2 = repair(G, child2)
    best_fit2, sol2 = filtering(G, repaired2)

    if best_fit1 > best_fit2:
        return best_fit1, sol1
    else:
        return best_fit2, sol2


# Main function
if __name__ == '__main__':
    # Read dataset number from command line arguments
    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset_number>")
        sys.exit(1)

    dataset_number = int(sys.argv[1])

    if dataset_number == 1:
        path = os.path.join(script_dir, "datasets", "first/")
        num_runs = 20
    elif dataset_number == 2:
        path = os.path.join(script_dir, "datasets", "second/")
        num_runs = 10
    else:
        print("Invalid dataset number. Please choose 1 or 2.")
        sys.exit(1)

    print(path)
    all_files = glob.glob(path + "*.txt")
    print(all_files)
    beta = 3 / 2
    Pa = 0.25
    sigma = pow(
        (gamma(1 + beta) * math.sin(math.pi * beta / 2)) / (gamma((1 + beta) / 2) * beta * pow(2, (beta - 1) / 2)),
        (1 / beta))

    output_path = os.path.join(path, "results", "results.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for filename in all_files:
            file_base_name = os.path.basename(filename)
            file_domination = []

            matrix = np.loadtxt(filename)
            G = nx.from_numpy_matrix(matrix, parallel_edges=False, create_using=None)
            G.remove_edges_from(nx.selfloop_edges(G))

            for run in range(num_runs):
                population = RandomPopulation(G)
                fitness_global = -9999999999
                x_best = []

                start_time = time.time()  # Record the start time

                for iteration in range(100):
                    for i in population:
                        i_index = population.index(i)
                        repaired = repair(G, i)
                        M_fitness, best = filtering(G, repaired)

                        x = i.copy()
                        Ry = random.choice(population)
                        y = Ry.copy()
                        repaired_y = repair(G, y)
                        best_fit_y, filtered_y = filtering(G, repaired_y)
                        z_fit, z_sol = crossover(G, x, y)

                        if z_fit > M_fitness:
                            population[i_index] = z_sol
                            M_fitness = z_fit

                        if fitness_global < M_fitness:
                            new_best = population[i_index].copy()
                            fitness_global = M_fitness
                            end_time = time.time()  # Record the end time
                            time_taken = end_time - start_time  # Calculate the time taken to find the solution
                            x_best.clear()
                            x_best.append(new_best)

                    for i in population:
                        j_index = population.index(i)
                        sum_x_best = sum(x_best[0])
                        u_compare = np.random.normal(0, 1)
                        if u_compare >= Pa:
                            worst = levy_flight(G, i, beta, sigma, sum_x_best)
                            repaired_worst = repair(G, worst)
                            best_filtering, sol = filtering(G, repaired_worst)

                            if fitness_global < best_filtering:
                                new_best = sol.copy()
                                fitness_global = best_filtering
                                x_best.clear()
                                x_best.append(new_best)
                                end_time = time.time()  # Record the end time
                                time_taken = end_time - start_time  # Calculate the time taken to find the solution

                file_domination.append((sum(x_best[0]), round(time_taken, 2)))
                print(sum(x_best[0]), time_taken)
            file_domination.append(file_base_name)
            print(file_domination)
            f.write("%s\n" % file_domination)
            f.write("\n")
    f.close()

