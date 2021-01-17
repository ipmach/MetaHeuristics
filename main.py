import matplotlib.pyplot as plt
from Problems.Parabola import Parabola
from Problems.Quadratic import Quadratic
from Problems.hardPolynomial import HardPolynomial
from Metaheuristics.local_search import LocalSearch
from Metaheuristics.simulated_anneling import SimmulatedAnneling
from Metaheuristics.iterated_local_search import IteratedLocalSearch
from Metaheuristics.genetic_algorithm import GeneticAlgorithm
from Metaheuristics.PBIL import PopulationBaseIncrementalLearning
from Metaheuristics.tabu_search import TabuSearch

#problem = Parabola()
#problem = Quadratic()
problem = HardPolynomial()
x, y = problem.generate_space()


#solver = LocalSearch(problem)
solver = SimmulatedAnneling(problem)
#solver = IteratedLocalSearch(problem, criterion='LSMC')
#solver = GeneticAlgorithm(problem, rank=False)
#solver = PopulationBaseIncrementalLearning(problem)
#solver = TabuSearch(problem)
solution, iterations = solver(max_iter=200)



plt.figure(figsize=(20, 5))
plt.subplot(1,2,1)
plt.title(solver.name)
plt.plot(x, y, label='Problem Space')
plt.plot(iterations, problem(iterations), '-o', label='Solutions')
plt.plot([solution], problem([solution]), 'o', label='Final Solution')
plt.legend()
plt.subplot(1,2,2)
plt.title("Cost Evolution")
plt.plot(problem(iterations))
plt.xlabel("Iterations")
plt.ylabel("Cost value")
plt.show()
