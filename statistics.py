import os
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import sys

def read_data(filename):
    data = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line:  # Ensure line is not empty
                items = eval(line)
                problem_name = items[-1]
                tuples = items[:-1]
                data[problem_name] = tuples
    return data

def calculate_statistics(data):
    solutions = [x[0] for x in data]
    times = [x[1] for x in data]

    min_solution = min(solutions)
    min_solution_with_min_time = None
    min_time = float('inf')

    max_solution = max(solutions)
    max_solution_with_max_time = None
    max_time = float('inf')

    for item in data:
        if item[0] == min_solution and item[1] < min_time:
            min_solution_with_min_time = item
            min_time = item[1]

        if item[0] == max_solution and item[1] < max_time:
            max_solution_with_max_time = item
            max_time = item[1]

    mean = np.mean(solutions)
    std_dev = round(np.std(solutions), 2)

    return min_solution_with_min_time, max_solution_with_max_time, min_time, max_time, mean, std_dev

def calculate_statistics_second(data):
    solutions = [x[0] for x in data]
    times = [x[1] for x in data]

    min_solution = min(solutions)
    min_solution_count = solutions.count(min_solution)
    min_solution_times = [time for sol, time in data if sol == min_solution]
    min_solution_min_time = min(min_solution_times) if min_solution_times else None

    max_solution = max(solutions)
    max_solution_with_max_time = None
    max_time = float('inf')

    mean = np.mean(solutions)
    std_dev = round(np.std(solutions), 2)

    return min_solution, min_solution_count, min_solution_min_time, mean, std_dev

def process_first_dataset():
    script_dir = os.path.dirname(__file__)
    data_file_path = os.path.join(script_dir, "datasets", "first", "results" , "results.txt")

    data = read_data(data_file_path)

    statistics = []
    for problem_name, tuples in data.items():
        min_solution, max_solution, min_time, max_time, mean, std_dev = calculate_statistics(tuples)
        statistics.append((problem_name, min_solution, max_solution, min_time, max_time, mean, std_dev))

    output_file_path = os.path.join(script_dir, "datasets", "first", "results" , "statistics.txt")
    with open(output_file_path, "w") as file:
        for stats in statistics:
            file.write(f"Instance: {stats[0]}\n")
            file.write(f"Best Solution: {stats[1]}\n")
            file.write(f"Worst Solution: {stats[2]}\n")
            file.write(f"Best Solution Time: {stats[3]}\n")
            file.write(f"Worst Solution Time: {stats[4]}\n")
            file.write(f"Mean: {stats[5]}\n")
            file.write(f"Standard Deviation: {stats[6]}\n\n")

def process_second_dataset():
    script_dir = os.path.dirname(__file__)
    data_file_path = os.path.join(script_dir, "datasets", "second", "results" , "results.txt")

    data = read_data(data_file_path)

    statistics = []
    for problem_name, tuples in data.items():
        min_solution, min_solution_count, min_solution_min_time, mean, std_dev = calculate_statistics_second(tuples)
        statistics.append((problem_name, min_solution, min_solution_count, min_solution_min_time, mean, std_dev))

    output_file_path = os.path.join(script_dir, "datasets", "second", "results" , "statistics.txt")
    with open(output_file_path, "w") as file:
        for stats in statistics:
            file.write(f"Instance: {stats[0]}\n")
            file.write(f"Smallest Solution: {stats[1]}\n")
            file.write(f"Reached: {stats[2]}\n")
            file.write(f"Best Time: {stats[3]}\n")
            file.write(f"Mean: {stats[4]}\n\n")

def plot_statistics():
    script_dir = os.path.dirname(__file__)
    data_file_path = os.path.join(script_dir, "datasets", "first", "results" , "statistics.txt")

    statistics = []
    with open(data_file_path, "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 8):
            instance_name = lines[i].split(": ")[1].strip()
            best_solution = eval(lines[i + 1].split(": ")[1].strip())
            best_solution_time = float(lines[i + 3].split(": ")[1].strip())
            statistics.append((instance_name, best_solution[0], best_solution_time))

    print(statistics)
    # x_values = [stats[1] for stats in statistics]
    # y_values = [stats[2] for stats in statistics]
    #
    # shortened_instance_names = [name.split('-')[0] + '-' + name.split('-')[-1] for name in [stats[0] for stats in statistics]]
    #
    # plt.figure(figsize=(10, 6))
    # plt.scatter(x_values, y_values, color='blue', label='Graph instance')
    #
    # texts = []
    # for i, instance_name in enumerate(shortened_instance_names):
    #     texts.append(plt.text(x_values[i], y_values[i], instance_name, fontsize=8))
    #
    # adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    #
    # degree = 2
    # coefficients = np.polyfit(x_values, y_values, degree)
    # polynomial = np.poly1d(coefficients)
    #
    # x_curve = np.linspace(min(x_values), max(x_values), 100)
    # y_curve = polynomial(x_curve)
    # plt.plot(x_curve, y_curve, color='red', label=f'Polynomial Fit (Degree {degree})')
    #
    # plt.xlabel('Best Solution Value')
    # plt.ylabel('Best Running Time')
    # plt.legend()
    #
    # plt.grid(True)
    # plt.savefig('runningTimes.png', bbox_inches='tight')
    # plt.show()

    x_values = [stats[1] for stats in statistics]  # Best solution value
    y_values = [stats[2] for stats in statistics]  # Best running time

    # Shorten instance names
    shortened_instance_names = [name.split('-')[0] + '-' + name.split('-')[-1] for name in
                                [stats[0] for stats in statistics]]

    # Plot scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, color='blue', label='Graph instance')

    # Create a list of text objects for annotation
    texts = []
    for i, instance_name in enumerate(shortened_instance_names):
        texts.append(plt.text(x_values[i], y_values[i], instance_name, fontsize=8))

    # Adjust text to minimize overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

    # Perform polynomial fitting
    degree = 2  # Change degree for polynomial fitting
    coefficients = np.polyfit(x_values, y_values, degree)
    polynomial = np.poly1d(coefficients)

    # Plot the polynomial curve
    x_curve = np.linspace(min(x_values), max(x_values), 100)
    y_curve = polynomial(x_curve)
    plt.plot(x_curve, y_curve, color='red', label=f'Polynomial Fit (Degree {degree})')

    # Add labels and legend
    plt.xlabel('Best Solution Value')
    plt.ylabel('Best Running Time')
    # plt.title('Scatter Plot with Polynomial Fit')
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.savefig('runningTimes.png', bbox_inches='tight')
    plt.show()

    print("Equation of the polynomial curve:")
    print(polynomial)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset>")
        print("Available datasets: first, second")
        sys.exit(1)

    dataset = sys.argv[1].lower()
    if dataset == "first":
        process_first_dataset()
        plot_statistics()
    elif dataset == "second":
        process_second_dataset()
    else:
        print("Invalid dataset specified. Available datasets: first, second")
        sys.exit(1)
