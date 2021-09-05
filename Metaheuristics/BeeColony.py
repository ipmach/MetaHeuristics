import numpy as np


class ABColony:

    def __init__(self, SN, D, lower_bound, upper_bound, fitness, abandon=30, iabc=False):
        """
        Metaheuristic algorithm of Bee colony
        :param SN: Number of scouts
        :param D: Internal variable
        :param lower_bound: Lower bound of the problem
        :param upper_bound: Upper bount of the problem
        :param fitness: fitness function
        :param abandon: Number of iterations to abandond flower
        :param iabc: Use improve version iabc
        """
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.fitness = fitness
        self.SN = SN
        self.D = D
        self.abandon = abandon
        self.iabc = iabc
        self.name = "Bee colony"
        self.abbreviation = 'abc'

    def new_flower(self):
        """
        New flower to jump
        :return: new flower
        """
        return self.lower_bound + np.random.rand(self.D) * (self.upper_bound - self.lower_bound)

    def initialize_population(self):
        """
        Initialize the population
        :return: new population
        """
        X = []
        self.abandond_list = np.zeros(self.SN)
        for i in range(self.SN):
            X.append(self.new_flower())
        self.f_first = np.array([self.fitness(x) for x in X])
        return X

    @staticmethod
    def theta():
        """
        Internal variable
        :return: int
        """
        return 2 * np.random.rand() - 1

    def W(self, X, i):
        """
        Internal variable
        :param X: population
        :param i: index population
        :return: int
        """
        return 1 / (1 + np.exp(-(self.fitness(X[i])) / self.f_first[i]))

    def theta2(self, X, i, employed=True):
        """
        Internal variable
        :param X: population
        :param i: index population
        :param employed: if is a employed bee
        :return: int
        """
        if employed:
            return 1
        return self.W(X, i)

    def calculate_V(self, X, i, j, k, employed=True):
        """
        Calculate value
        :param X: population
        :param i: index population 1
        :param j: index of value in population
        :param k: index population 2
        :param employed: if the bee is employed
        :return: value
        """
        if self.iabc:
            theta_ = ABColony.theta()
            return X[i][j] * self.W(X, i) + 2 * (theta_ - 0.5) * (X[i][j] - X[k][j]) * \
                   self.W(X, i) + theta_ * (X[i][j] - X[k][j]) * self.theta2(X, i, employed=employed)
        return X[i][j] + ABColony.theta() * (X[i][j] - X[k][j])

    def employed_bee(self, X):
        """
        Employed bee
        :param X: population
        :return: new population
        """
        for i in range(self.SN):
            j = np.random.choice(self.D)
            k = np.random.choice(self.SN)
            V = X[i].copy()
            while k == i:
                k = np.random.choice(self.SN)
            V[j] = self.calculate_V(X, i, j, k)
            if self.fitness(X[i]) > self.fitness(V) and np.all(self.lower_bound <= V <= self.upper_bound):
                X[i] = V.copy()
                self.abandond_list[i] = 0
            else:
                self.abandond_list[i] += 1
            if self.abandond_list[i] == self.abandon:
                # print("Abandond flower", i)
                self.abandond_list[i] = 0
                X[i] = self.new_flower()
        return X

    def scout_bee(self, X):
        """
        Scout bee
        :param X: population
        :return: best candidate and fitness
        """
        f = np.array([self.fitness(x) for x in X]).reshape(-1)
        if not np.all(f >= 0):
            f = np.min(f) + f
        P = f / np.sum(f)
        idx = np.random.choice(np.arange(len(X)), p=P)
        return X[idx], f[idx]

    def __call__(self, max_iter=100, initial_P=None):
        """
        Solve problem
        :param max_iter: maximun number of iterations
        :param initial_P: if we have a initial population (Default None)
        :return: Solution, history
        """
        if initial_P is None:
            X = self.initialize_population()
        else:
            X = initial_P.copy()
        solution, value = self.scout_bee(X)
        history = [solution]
        solutions_h = [value]
        for _ in range(max_iter):
            X = self.employed_bee(X)
            solution, value = self.scout_bee(X)
            history.append(solution)
            solutions_h.append(value)
        f = np.argmin([self.fitness(x) for x in history])
        return history[f], history
