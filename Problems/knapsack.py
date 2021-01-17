import numpy as np


class KSP:
    
    def __init__(self, size, max_weight=100, min_weight=0, 
                 max_value=100, min_value=0, per_remove=0.2):
        self.size = size
        self.jump = 0
        self.x = np.arange(size)
        self.w = np.random.choice(np.arange(min_weight,
                                            max_weight), size)
        self.v = np.random.choice(np.arange(min_value,
                                            max_value), size)
        w_v = list(zip(self.w, self.v))
        original = len(w_v) * (1 - per_remove)
        while(len(w_v) > original):
            index = np.random.choice(np.arange(len(w_v)))
            w_v.pop(index)
        _, v = list(zip(*w_v))
        self.total = np.sum(v)

    def generate_space(self):
        return self.x, self.w, self.v, self.total

    def get_incremental(self):
        w = [0]
        v = [0]
        for i in range(len(self.x)):
             w.append(w[-1] + self.w[i])
             v.append(v[-1] + self.v[i])
        return self.x, w, v, self.total

    def __call__(self, x, real=False):
        v = np.array(self.v)[x]
        if real:
            return np.sum(v)
        return self.total - np.sum(v)

    def give_random_point(self):
        w_v = list(zip(self.w, self.v))
        aux = np.sum(self.w)
        x_ = self.x.copy()
        while aux > self.total:
            size = np.random.randint(5, self.size)
            x_ = np.random.choice(self.x, size, replace=False)
            w_v_ = [w_v[i] for i in x_]
            w, _ = zip(*w_v_)
            aux = np.sum(list(w))
        return x_

    def perturbation(self, solution, p=2):
        x = list(self.x.copy())
        for s in solution:
            x.remove(s)
        while(True):
            solution = list(solution)
            s = np.random.choice(solution)
            update_w = np.sum(self.w[solution]) - self.w[s]
            solution.remove(s)
            np.random.shuffle(x)
            for _ in x:
                for i in x:
                    if update_w + self.w[i] <= self.total:
                        solution.append(i)
                        return solution

    def generate_neighbour(self, x):
        return self.perturbation(x)

    def all_neighbours(self, x):
        neighbours = []
        for i in range(10):
            neighbours.append(self.generate_neighbour(x))
        return neighbours



