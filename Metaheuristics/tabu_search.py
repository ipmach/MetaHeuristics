import numpy as np


class TabuSearch:

    def __init__(self, problem):
        self.problem = problem
        self.name = "Tabu Search"
        self.abbreviation = 'TB'

    def in_list(self, c, clist):
        """
        Check if is in the list
        :param c: candidate
        :param clist: list
        :return:
        """
        for cl in clist:
            if isinstance(c, list) or type(c).__module__ == np.__name__:
                if len(c) == len(cl) and all(c == cl):
                    return True
            else:
                if c == cl:
                    return True
        return False

    def __call__(self, max_iter=1000, list_size=40,
                 num_candidates=4):
        """
        Solver
        :param max_iter: max number of iterations
        :param list_size: max size of tabu list
        :param num_candidates: number of candidates generated
        :return: Solution and history of solutions
        """
        solution = self.problem.give_random_point()
        history = [solution]
        tabulist = [solution]
        for _ in range(max_iter):
            candidates = [self.problem.give_random_point()
                          for _ in range(num_candidates)]
            bestCandidate = candidates[0]
            f_best = self.problem(bestCandidate)
            for c in candidates:
                if self.in_list(c, tabulist) and self.problem(c) < f_best:
                    bestCandidate = c
                    f_best = self.problem(bestCandidate)
            if f_best < self.problem(solution):
                solution = bestCandidate
                history.append(solution)
            tabulist.append(bestCandidate)
            if len(tabulist) > list_size:
                tabulist = tabulist[1:]
        return solution, history