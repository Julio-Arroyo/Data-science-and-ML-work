import matplotlib.pyplot as plt
import numpy as np
import random
import math


def b(epsilon, N):
    return 2 * math.exp(-2 * math.pow(epsilon, 2) * N)


def plot_hoeffding(distributions):
    epsilons = np.arange(0, 1, 0.01)
    nus = list(range(11))  # divide nus[i] by 10 to get actual nu
    mu = 0.5
    Ps = [list(), list(), list()]
    for i in range(len(distributions)):
        for epsilon in epsilons:
            num_true_conditions = 0
            for nu in nus:
                if abs(nu - mu) > epsilon and nu / 10 in distributions[i]:
                    num_true_conditions += distributions[i][nu / 10]
            Ps[i].append(num_true_conditions / 1000)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    B = np.vectorize(b)
    fig.suptitle('P(epsilon)')
    ax1.plot(epsilons, Ps[0])
    ax1.plot(epsilons, B(epsilons, 1000))
    ax2.plot(epsilons, Ps[1])
    ax2.plot(epsilons, B(epsilons, 1000))
    ax3.plot(epsilons, Ps[2])
    ax3.plot(epsilons, B(epsilons, 1000))
    plt.show()


def generate_distributions():
    sum_nu_min = 0
    nu_1_distr = {}
    nu_rand_distr = {}
    nu_min_distr = {}
    for i in range(100000):
        if i % 5000 == 0:
            print('{}%'.format(i/1000))
        
        c_rand = random.randint(0,999)
        nu_rand = None

        nu_min = float('inf')
        c_min = None

        nu_1 = None
        for j in range(1000):
            heads = 0
            for k in range(10):
                if random.random() < 0.5:
                    heads += 1
            if j == 0:
                nu_1 = heads / 10
                if not nu_1 in nu_1_distr:
                    nu_1_distr[nu_1] = 1
                else:
                    nu_1_distr[nu_1] += 1
            if j == c_rand:
                nu_rand = heads / 10
                if not nu_rand in nu_rand_distr:
                    nu_rand_distr[nu_rand] = 1
                else:
                    nu_rand_distr[nu_rand] += 1
            if heads / 10 < nu_min:
                nu_min = heads / 10
        if not nu_min in nu_min_distr:
            nu_min_distr[nu_min] = 1
        else:
            nu_min_distr[nu_min] += 1
        sum_nu_min += nu_min
    return(sum_nu_min, nu_1_distr, nu_rand_distr, nu_min_distr)


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


def linear_regression(dataset):
    X = []
    y = []
    for i in range(len(dataset)):
        X.append(dataset[i][0])
        y.append(dataset[i][1])
    X = np.array(X)
    X_T = X.transpose()
    y = np.array(y)
    X_pi = np.matmul(np.linalg.inv(np.matmul(X_T, X)), X_T)
    w = np.matmul(X_pi, y)
    g = get_line((-w[0]/w[1],0), (0, -w[0]/w[2]))
    return g


def get_error(f, g, dataset):
    disagreements = 0
    for i in range(len(dataset)):
        datapoint = (dataset[i][0])
        if not evaluate_point(datapoint, f) == evaluate_point(datapoint, g):
            disagreements += 1
    return disagreements / len(dataset)


if __name__ == '__main__':
    # nu_1_distr = {0.0: 83, 0.1: 942, 0.2: 4410, 0.3: 11778, 0.4: 20415, 0.5: 24526, 0.6: 20536,  0.7: 11718, 0.8: 4506,  0.9: 995,  1.0: 91}
    # nu_rand_distr = {0.0: 113, 0.1: 1002, 0.2: 4585, 0.3: 11729, 0.4: 20246,  0.5: 24852,  0.6: 20319, 0.7: 11749, 0.8: 4347, 0.9: 960, 1.0: 98}
    # nu_min_distr = {0.0: 62454, 0.1: 37543, 0.2: 3, 0.3:0, 0.4:0, 0.5:0, 0.6:0, 0.7:0, 0.8:0, 0.9:0, 1:0}

    # Hoeffding Inequality
    # (sum_nu_min, nu_1_distr, nu_rand_distr, nu_min_distr) = generate_distributions()
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # print('Average nu_min: {}'.format(sum_nu_min / 100000))
    # print('nu_1 distribution {}'.format(nu_1_distr))
    # print('nu_rand distribution {}'.format(nu_rand_distr))
    # print('nu_min distribution {}'.format(nu_min_distr))
    # fig.suptitle('Histograms nu_1, nu_rand, nu_min')
    # ax1.bar(nu_1_distr.keys(), nu_1_distr.values(), tick_label = [str(v) for v in nu_1_distr.keys()])
    # ax2.bar(nu_rand_distr.keys(), nu_rand_distr.values())
    # ax3.bar(nu_min_distr.keys(), nu_min_distr.values())
    # distributions = [nu_1_distr, nu_rand_distr, nu_min_distr]
    # plot_hoeffding(distributions)

    # Linear regression
    sum_E_in = 0
    sum_E_out = 0
    learned_lines = []
    for i in range(1000):
        (f, dataset) = generate_dataset(100)
        (F, random_data_set) = generate_dataset(100)
        g = linear_regression(dataset)
        learned_lines.append(g)
        E_in = get_error(f, g, dataset)
        E_out = get_error(f, g, random_data_set)
        sum_E_in += E_in
        sum_E_out += E_out
    E_in_avg = sum_E_in / 1000
    print('Average in sample error {}'.format(E_in_avg))
    E_out_avg = sum_E_out / 1000
    print('Average out of sample error {}'.format(E_out_avg))
        
        

