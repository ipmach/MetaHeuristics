import matplotlib.pyplot as plt
from Problems.hardPolynomial import HardPolynomial
from Metaheuristics.local_search import LocalSearch
from Metaheuristics.simulated_anneling import SimmulatedAnneling
from Metaheuristics.iterated_local_search import IteratedLocalSearch
from Metaheuristics.genetic_algorithm import GeneticAlgorithm
from Metaheuristics.PBIL import PopulationBaseIncrementalLearning
from Metaheuristics.tabu_search import TabuSearch
from Metaheuristics.BeeColony import ABColony
from tqdm import tqdm
import numpy as np



problem = HardPolynomial()
x, y = problem.generate_space()


sol1 = []
sol2 = []
sol3 = []
sol4 = []
sol5 = []
sol6 = []
sol7 = []
sol8 = []
sol9 = []
sol10 = []
sol11 = []

for _ in tqdm(range(100)):
    solver1 = LocalSearch(problem)
    solver2 = SimmulatedAnneling(problem)
    solver3 = IteratedLocalSearch(problem, criterion='LSMC')
    solver4 = IteratedLocalSearch(problem, criterion='better')
    solver5 = IteratedLocalSearch(problem, criterion='rw')
    solver6 = GeneticAlgorithm(problem, rank=False)
    solver7 = GeneticAlgorithm(problem, rank=True)
    solver8 = PopulationBaseIncrementalLearning(problem)
    solver9 = TabuSearch(problem)
    solver10 = ABColony(20, 1, -10, 10, problem, iabc=False)
    solver11 = ABColony(20, 1, -10, 10, problem, iabc=True)
    solution1, _ = solver1()
    solution2, _ = solver2()
    solution3, _ = solver3()
    solution4, _ = solver4()
    solution5, _ = solver5()
    solution6, _ = solver6()
    solution7, _ = solver7()
    solution8, _ = solver8()
    solution9, _ = solver9()
    solution10, _ = solver10()
    solution11, _ = solver11()
    sol1.append(solution1)
    sol2.append(solution2)
    sol3.append(solution3)
    sol4.append(solution4)
    sol5.append(solution5)
    sol6.append(solution6)
    sol7.append(solution7)
    sol8.append(solution8)
    sol9.append(solution9)
    sol10.append(solution10)
    sol11.append(solution11)

names = [solver1.abbreviation, solver2.abbreviation,
         solver3.abbreviation + '-L', solver4.abbreviation + '-B',
         solver5.abbreviation + '-R', solver6.abbreviation,
         solver7.abbreviation + '-r', solver8.abbreviation,
         solver9.abbreviation, solver10.abbreviation, solver11.abbreviation + '-iabc']
solution = [np.mean(problem(sol1)), np.mean(problem(sol2)),
            np.mean(problem(sol3)),
            np.mean(problem(sol4)), np.mean(problem(sol5)),
            np.mean(problem(sol6)), np.mean(problem(sol7)),
            np.mean(problem(sol8)), np.mean(problem(sol9)),
            np.mean(problem(sol10)), np.mean(problem(sol11))]

plt.figure(figsize=(16,5))
plt.bar(names, solution)
plt.title("Performance")
plt.xlabel("Metaheuristics")
plt.ylabel("Cost Value")
plt.show()