from Problems.knapsack import KSP
from Metaheuristics.local_search import LocalSearch
from Metaheuristics.simulated_anneling import SimmulatedAnneling
#from Metaheuristics.genetic_algorithm import GeneticAlgorithm
from Metaheuristics.tabu_search import TabuSearch
from Metaheuristics.iterated_local_search import IteratedLocalSearch



import matplotlib.pyplot as plt
import numpy as np


problem = KSP(500)

#solver = LocalSearch(problem)
solver = SimmulatedAnneling(problem)
#solver = IteratedLocalSearch(problem, criterion='LSMC')
#solver = TabuSearch(problem)


x, w, v, t = problem.get_incremental()
solution, iterations = solver(max_iter=200)
iterations = [problem(i, real=True) for i in iterations]
iter = np.arange(len(iterations))


plt.figure(figsize=(12,9))
plt.subplot(2,2,1)
plt.plot(v, label='value')
plt.plot(w, label='weight')
plt.plot(np.ones(len(w)) * t, label='max')
plt.ylabel("Value/weight")
plt.title('KSP problem')
plt.legend()
plt.subplot(2,2,2)
plt.title('Fitness')
plt.plot(iter, iterations, '-o', label='Solutions')
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.subplot(2,2,3)
plt.title('Solution vs Max')
plt.bar(["Value", "Weight"],
        [problem(problem.x, real=True),
         problem.total],
         label='Max')
plt.bar(["Value", "Weight"],
        [problem(solution, real=True),
         np.sum(problem.w[solution])],
         label='Solution')
plt.legend()
plt.subplot(2,2,4)
plt.scatter(problem.v, problem.w, label='all')
v = [problem.v[i] for i in solution]
w = [problem.w[i] for i in solution]
plt.scatter(v, w, label='solution')

plt.title('Problem space')
plt.xlabel('Value')
plt.ylabel('Weight')
plt.legend()
plt.show()

