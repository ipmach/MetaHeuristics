from Encode.binaryCode import BinaryCode
import numpy as np

class PopulationBaseIncrementalLearning:

    def __init__(self, problem):
        self.problem = problem
        self.bc = BinaryCode(jump=problem.jump)
        self.name = "Population Base Incremental Learning"
        self.abbreviation = 'PBIL'

    def generate(self, P, generation_number):
        """
        Generate population form Array P
        :param P: array P
        :param generation_number: Population size generated
        :return: Population generated
        """
        population = []
        for _ in range(generation_number):
            aux = [str(np.random.choice([0, 1], p=[1-p,p]))
                                        for p in P]
            aux = "".join(aux)
            while not(self.problem.min_ <= self.bc.bin2decv(aux) <= self.problem.max_):
                aux = [str(np.random.choice([0, 1], p=[1 - p, p]))
                       for p in P]
                aux = "".join(aux)
            population.append(aux)
        return population

    def __call__(self, max_iter=1000, learning_rate=0.01,
                 generation_number=5, number_update=2):
       """
        Solver
        :param max_iter: max number of iterations
        :param learning_rate: learning rate
        :param generation_number: Population size generated
        :param number_update: population use to update P
        :return: Solution and history of solutions
       """
       max_value = (1/self.problem.jump) * (self.problem.max_ -
                                            self.problem.min_)
       # Initialize P
       P = np.ones(len(self.bc.dec2bin(max_value))) * 0.5
       population = self.generate(P, generation_number)
       eval = self.problem(self.bc.bin2decv(population))
       history = [population[np.argmin(eval)]]
       for _ in range(max_iter):
           pop_eva = list(zip(population, eval))
           pop_eva.sort(key=lambda x: x[1])
           population, _ = zip(*pop_eva)
           for i in range(number_update):
               aux = np.array([int(i) for i in list(population[i])])
               P = (1-learning_rate) * P + learning_rate * aux
           population = self.generate(P, generation_number)
           eval = self.problem(self.bc.bin2decv(population))
           history.append(population[np.argmin(eval)])
       solution = self.bc.bin2decv(population[np.argmin(eval)])
       history = self.bc.bin2decv(history)
       return solution, history
