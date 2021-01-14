import copy
import numpy as np
from Metaheuristics.local_search import LocalSearch
import random


class IteratedLocalSearch:

    def __init__(self, problem, criterion='better'):
        self.problem = problem
        self.name = "Iterated Local Search"
        self.abbreviation = 'ILS'
        if criterion=='better':
            self.AcceptanceCriterion = self.better
        elif criterion=='LSMC':
            self.AcceptanceCriterion = self.LSMC
        else:
            self.AcceptanceCriterion = self.rw

    def perturbation(self, solution, p=4):
        """
        Perturbation to avoid local optimun
        :param solution: actual solution
        :param p: perturbation change
        :return: new solution
        """
        if solution + p > self.problem.max_:
            return solution - p

        if solution - p < self.problem.min_:
            return solution + p

        return np.random.choice([solution - p, solution + p])

    def better(self, solution, solution_):
        """
        Acceptation criterion
        :param solution: old solution
        :param solution_: new solution
        :return: solution decide
        """
        if self.problem(solution_) < self.problem(solution):
            return solution_
        return solution

    def rw(self, solution, solution_):
        """
        Acceptation criterion
        :param solution: old solution
        :param solution_: new solution
        :return: solution decide
        """
        return solution_

    def LSMC(self, solution, solution_, T=0.5):
        """
        Acceptation criterion
        :param solution: old solution
        :param solution_: new solution
        :return: solution decide
        """
        if self.problem(solution_) < self.problem(solution):
            return solution_
        cost = self.problem(solution) - self.problem(solution_)
        if random.uniform(0, 1) <= np.exp(cost/T):
            return solution_
        return solution

    def __call__(self, max_iter=1000):
        """
        Solver
        :param max_iter: max number of iterations
        :return: Solution and history of solutions
        """
        localsearch = LocalSearch(self.problem)
        solution = self.problem.give_random_point()
        solution, _ = localsearch(solution=solution)
        history = [solution]
        break_ = 0
        for _ in range(max_iter):
            solution_ = self.perturbation(solution)
            solution_, _ = localsearch(solution=solution_)
            solution = self.AcceptanceCriterion(solution, solution_)
            history.append(copy.copy(solution))
            if solution - history[-2] <= 10e-9:
                break_ += 1
            else:
                break_ = 0
            if break_ > 4: # stop algorithm
                break
        # Take best solution
        #solution = history[np.argmin(self.problem(history))]
        return solution, history
