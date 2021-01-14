from Encode.binaryCode import BinaryCode
import numpy as np
import copy

class GeneticAlgorithm:

    def __init__(self, problem, elit_=True, rank=False):
        self.problem = problem
        self.bc = BinaryCode(jump=problem.jump)
        self.name = "Genetic Algorithm"
        self.abbreviation = 'GA'
        self.do_rank = rank
        if elit_:
            self.reconstruction = self.elit
        else:
            self.reconstruction = self.noElite

    def generatePopulation(self, population_size):
        """
        Generate a new population
        :param population_size: Size of the population
        :return: generated population encoded
        """
        return self.bc.dec2binv([self.problem.give_random_point()
                                 for _ in range(population_size)])

    def crossover(self, couples, population_size):
        """
        Crossover of couples
        :param couples: couples list
        :param population_size: population size
        :return: childs generated
        """
        childs = []
        for i, j in couples:
            mid = np.random.choice(np.arange(len(i)))
            childs.append(i[:mid] + j[mid:])
            childs.append(j[:mid] + i[mid:])
        return childs[:population_size]

    def rank(self, fitness):
        """
        Rank the fitness
        :param fitness: fitness values
        :return: fitness updated
        """
        P_w = np.min(fitness)
        P_b = np.max(fitness)
        N = len(fitness)
        aux = np.zeros(N)
        for i in range(aux.shape[0]):
            aux[i] = (1 / N) * (P_w + (P_b - P_w) * \
                                ((fitness[i] - 1) / (N - 1)))
        return aux


    def selection(self, parents_num, eval, population):
        """
        Selection parents
        :param parents_num: number of parents
        :param eval: evaluation population
        :param population: population list
        :return: parent list
        """
        parents = []
        if self.do_rank:
            aux = self.rank(eval)
        aux = np.max(eval) - np.array(eval)
        prob = aux / np.sum(aux)
        for _ in range(parents_num):
            parents.append(np.random.choice(population,
                                            p=prob, size=2))
        return parents

    def mutation(self, population, mutation=0.01):
        """
        Generate mutations in the population
        :param population: list of the population
        :param mutation: mutation rate
        :return: mutated population
        """
        new_population = []
        for p in population:
            n = len(p)
            d = np.random.choice([0, 1], p=[1 - mutation,
                                 mutation], size=n)
            aux = copy.copy(p)
            for j,di in enumerate(d):
                if di:
                   aux = list(aux)
                   aux[j] = str(int(not int(aux[j])))
                   aux = "".join(aux)
            new_population.append(copy.copy(aux))
        return new_population

    def noElite(self, population, childs):
        """
        Reconstruction with no elitism
        :param population: list population
        :param childs: list childs
        :return: new population
        """
        return childs

    def elit(self, population, childs):
        """
        Reconstruction with elitism
        :param population: list population
        :param childs: list childs
        :return: new population
        """
        total = list(childs) + list(population)
        eval = self.problem(self.bc.bin2decv(total))
        total_eval = list(zip(total, eval))
        total_eval.sort(key=lambda x: x[1])
        total, _ = zip(*total_eval)
        total = list(total)
        return total[:len(population)]

    def __call__(self, max_iter=1000, population_size=6,
                 parents_num=4):
        """
        Solver
        :param max_iter: max number of iterations
        :param population_size: Size population
        :param parents_num: Number of parents
        :return: Solution and history of solutions
        """
        population = self.generatePopulation(population_size)
        eval = self.problem(self.bc.bin2decv(population))
        history = [population[np.argmin(eval)]]
        for _ in range(max_iter):
            parents = self.selection(parents_num, eval, population)
            childs = self.crossover(parents, population_size)
            childs = self.mutation(childs)
            population = self.reconstruction(population, childs)
            eval = self.problem(self.bc.bin2decv(population))
            history.append(population[np.argmin(eval)])
            if np.sum(np.max(eval) - np.array(eval)) == 0:
                break
        solution = self.bc.bin2decv(population[np.argmin(eval)])
        history = self.bc.bin2decv(history)
        return solution, history