import copy
import numpy as np
import random

class SimmulatedAnneling:

    def __init__(self, problem, reduction='geometric'):
        self.problem = problem
        self.name = "Simulated Anneling"
        self.abbreviation = 'SA'
        if reduction == 'geometric':
            self.reduction = self.geometric_temperature
        else:
            self.reduction = self.linear_reduction

    def geometric_temperature(self, T, c):
        """
        Change temperature in a geometric function
        :param T: Temperature
        :param c: cooling factor
        :return: new temperature
        """
        return T * c

    def linear_reduction(self, T, c):
        """
        Change temperature in a linear function
        :param T: Temperature
        :param c: cooling factor
        :return: new temperature
        """
        return T - c

    def __call__(self, solution=None, max_iter=1000, T=90,
                 c=0.9, threshold=0.001):
        """
        Solver
        :param solution: initial solution
        :param max_iter: max number of iterations
        :param T: Initial temperature
        :param c: Cooling factor
        :param threshold: break threshold
        :return: Solution and history of solutions
        """
        if solution is None:
            solution = self.problem.give_random_point()
        history = [solution]
        for _ in range(max_iter):
            solution_ = self.problem.give_random_point()
            cost = self.problem(solution_) - self.problem(solution)
            if random.uniform(0, 1) <= np.exp(-cost/T):
                history.append(copy.copy(solution_))
                solution = solution_

            T = self.reduction(T, c)
            if T < threshold:
                break
        return solution, history