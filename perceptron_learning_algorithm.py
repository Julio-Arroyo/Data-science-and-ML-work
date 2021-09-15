import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.lib import twodim_base
from scipy import integrate


def get_rand_num():
    return random.random() * 2 - 1

def get_line(p1, p2):
    m = (p2[1] - p1[1])/(p2[0] - p1[0])
    b = (p2[1] - m * p2[0])
    return (m, b)


def get_random_line():
    p1 = (get_rand_num(), get_rand_num())
    p2 = (get_rand_num(), get_rand_num())
    return get_line(p1, p2)


def evaluate_point(x_i, line):
    if x_i[2] < line[0] * x_i[1] + line[1]:
        return -1
    else:
        return 1


def generate_dataset(N):
    line = get_random_line()
    dataset = []
    for _ in range(N):
        x_i = np.array([1, get_rand_num(), get_rand_num()])
        dataset.append((x_i, evaluate_point(x_i, line)))
    return (line, dataset)


def sign(num):
    if num == 0:
        return 0
    elif num < 0:
        return -1
    else:
        return 1


def get_misclassified_pts(w, dataset):
    misclassified = []
    for datapoint in dataset:
        if not sign(np.inner(w, datapoint[0])) == datapoint[1]:
            misclassified.append(datapoint)
    return misclassified


def pla(dataset):
    """
    Perceptron Learning Algorithm (PLA). Find a function to classify correctly all
    points in dataset.
    """
    w = np.zeros(3)
    num_iterations = 0
    while True:
        misclassified = get_misclassified_pts(w, dataset)
        if len(misclassified) == 0:
            break
        datapoint = misclassified.pop(random.randrange(len(misclassified)))
        w = w + datapoint[1] * datapoint[0]
        num_iterations += 1
    g = get_line((-w[0]/w[1],0), (0, -w[0]/w[2]))
    return (g, num_iterations)


def get_disagreement(f, g, N):
    disagreements = 0
    for _ in range(N):
        rand_point = (1, get_rand_num(), get_rand_num())
        if not evaluate_point(rand_point, f) == evaluate_point(rand_point, g):
            disagreements += 1
    return disagreements / N


def calc_left_area(v0, v1, f, g):
    if v0 >= -1:
        side = v1 - v0
        height = abs(f[0] * v1 + f[1] - (g[0] * v1 + g[1]))
        return 0.5 * side * height
    else:
        side = v1 - (-1)
        h1 = abs(f[0] * v1 + f[1] - (g[0] * v1 + g[1]))
        y1 = f[0] * v0 + f[1]
        y2 = g[0] * v0 + g[1]
        h2 = None
        if abs(y1) < abs(y2):
            h2 = abs(sign(y2) - y1)
        else:
            h2 = abs(sign(y1) - y2)
        return ((h1 + h2) / 2) * side


def calc_right_area(v2, v3, f, g):
    if v2 <= 1:
        side = v3 - v2
        height = abs(f[0] * v2 + f[1] - (g[0] * v2 + g[1]))
        return 0.5 * side * height
    else:
        side = 1 - v2
        h1 = abs(f[0] * v2 + f[1] - (g[0] * v2 + g[1]))
        y1 = f[0] * v3 + f[1]
        y2 = g[0] * v3 + g[1]
        h2 = None
        if abs(y1) < abs(y2):
            h2 = abs(sign(y2) - y1)
        else:
            h2 = abs(sign(y1) - y2)
        return ((h1 + h2) / 2) * side
            

def get_exact_disagreement(f, g):
    area = 0
    values = []
    f_up = (1 - f[1]) / f[0]
    f_down = (-1 - f[1]) / f[0]
    g_up = (1 - g[1]) / g[0]
    g_down = (-1 - g[1]) / g[0]
    values.append(f_up)
    values.append(f_down)
    values.append(g_up)
    values.append(g_down)
    values.sort()
    integral_in_square = (f_up <=-1 or f_up>=1) and (g_up <=-1 or g_up>=1) and (f_down <=-1 or f_down>=1) and (g_down <= -1 or g_down >= 1)
    h = lambda x: abs(f[0]*x + f[1] - (g[0]*x + g[1]))
    if abs(values[1]) <= 1 and abs(values[2]) <= 1:
        area += abs(integrate.quad(h, values[1], values[2])[0])
        area += calc_left_area(values[0], values[1], f, g)
        area += calc_right_area(values[2], values[3], f, g)
    elif abs(values[1]) <= 1:
        area += abs(integrate.quad(h, values[1], 1)[0])
        area += calc_left_area(values[0], values[1], f, g)
    elif abs(values[2]) <= 1:
        area += abs(integrate.quad(h, -1, values[2])[0])
        area += calc_right_area(values[2], values[3], f, g)
    else:
        area += abs(integrate.quad(h, -1, 1)[0])
    return (area / 4.0, area)


def visualize(dataset, f, g):
    fig = plt.figure()
    ax = plt.axes()
    x_vals1 = []
    y_vals1 = []
    x_vals0 = []
    y_vals0 = []
    for datapoint in dataset:
        if datapoint[1] == 1:
            x_vals1.append(datapoint[0][1])
            y_vals1.append(datapoint[0][2])
        else:
            x_vals0.append(datapoint[0][1])
            y_vals0.append(datapoint[0][2])
    x = np.linspace(-1, 1, 500)
    plt.plot(x, f[0]*x + f[1], '--c')
    plt.plot(x, g[0]*x + g[1], '-.r')
    plt.scatter(x_vals1, y_vals1)
    plt.scatter(x_vals0, y_vals0)
    plt.show()


if __name__ == '__main__':
    # TEST evaluate line
    # (f, dataset) = generate_dataset(100)
    # (g, iterations) = pla(dataset)
    # visualize(dataset, f, g)

    # TEST calculate disagreement exactly
    # print(get_exact_disagreement((2, 0.2), (1.9, 0.22)))
    # mistakes = 0
    # for _ in range(1000):
    #     f = get_random_line()
    #     if abs(f[1]/f[0]) >= 1:
    #         continue
    #     g = get_random_line()
    #     if abs(g[1]/g[0]) >= 1:
    #         continue
    #     exact = get_exact_disagreement(f, g)[0]
    #     estimate = get_disagreement(f, g, 60000)
    #     assert abs(exact - estimate) <= 0.02, 'f: {}. g: {}. Exact = {}. Estimate = {}.'.format(f, g, exact, estimate)
    # print(mistakes)


    # ACTUAL EXPERIMENT
    sum_iterations = 0
    sum_disagreements = 0
    for i in range(1000):
        (f, dataset) = generate_dataset(10)
        (g, iterations) = pla(dataset)
        sum_iterations += iterations
        disagreement = get_disagreement(f, g, 100000)
        sum_disagreements += disagreement
        if i % 50 == 0:
            print('{}%'.format(i/10))
            visualize(dataset, f, g)
    avg_iterations = sum_iterations / 1000
    avg_disagreement = sum_disagreements / 1000
    print('Average iterations: {}. Average disagreement: {}'.format(avg_iterations, avg_disagreement))
