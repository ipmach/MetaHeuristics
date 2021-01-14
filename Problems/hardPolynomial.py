import numpy as np


class HardPolynomial:

    def __init__(self, jump=0.01, min_=-10, max_=10):
        self.jump = jump
        self.min_ = min_
        self.max_ = max_
        self.f = lambda x: 30 * np.sin(x) - x ** 2 + 120
        self.fv = np.vectorize(self.f)

    def __call__(self, x):
        return self.fv(x)

    def give_random_point(self):
        return np.round(np.random.choice(np.arange(self.min_, self.max_,
                                                   self.jump)), 1)

    def generate_space(self):
        x = np.arange(self.min_, self.max_, self.jump)
        y = self.fv(np.arange(self.min_, self.max_,
                              self.jump))
        return x, y

    def generate_neighbour(self, x):
        if x < self.min_ and x > self.max_:
            return None

        if x - self.jump < self.min_:
            return x + self.jump

        if x + self.jump > self.max_:
            return x - self.jump

        return np.random.choice([x - self.jump,
                                 x + self.jump])

    def all_neighbours(self, x):
        if x < self.min_ and x > self.max_:
            return None

        if x - self.jump < self.min_:
            return [x + self.jump]

        if x + self.jump > self.max_:
            return [x - self.jump]

        return [x + self.jump, x - self.jump]