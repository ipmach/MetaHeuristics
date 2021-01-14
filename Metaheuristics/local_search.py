import copy

class LocalSearch:

    def __init__(self, problem):
        self.problem = problem
        self.name = "Local search"
        self.abbreviation = 'LS'

    def __call__(self, solution=None, max_iter=1000):
        """
        Solver
        :param max_iter: max number of iterations
        :param solution: solution defined
        :return: Solution and history of solutions
        """
        if solution is None:
            solution = self.problem.give_random_point()
        history = [solution]
        for _ in range(max_iter):
            solution_ = self.problem.generate_neighbour(solution)
            if self.problem(solution_) < self.problem(solution):
                history.append(copy.copy(solution_))
                solution = solution_
            f_s = self.problem(solution)
            test = [f_s <= self.problem(i) for i in
                    self.problem.all_neighbours(solution)]
            if all(test):
                break
        return solution, history
