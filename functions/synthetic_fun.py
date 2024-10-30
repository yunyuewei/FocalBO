import numpy as np


class Rastrigin:
    def __init__(self, dim=10, minimize=True):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.name = 'Rastrigin'
        self.minimize = minimize

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        f = 10*self.dim + np.sum(x**2-np.cos(2*np.pi*x))
        if not self.minimize:
            return -f
        else:
            return f


class Ackley:
    def __init__(self, dim=10, minimize=True):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.name = 'Ackley'
        self.minimize = minimize

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        # print(x)
        assert np.all(x <= self.ub) and np.all(x >= self.lb), x
        a, b, c = 20.0, 0.2, 2*np.pi
        f = -a*np.exp(-b*np.sqrt(np.mean(x**2)))
        f -= np.exp(np.mean(np.cos(c*x)))
        f += a + np.exp(1)
        if not self.minimize:
            return -f
        else:
            return f


class Levy:
    def __init__(self, dim=1, minimize=True):
        assert dim > 0
        self.dim = dim
        self.minimize = minimize
        self.lb = -5 * np.ones(dim)
        self.ub = 5 * np.ones(dim)
        self.name='Levy'

    def __call__(self, x):
        x = np.array(x)
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        w = []
        for idx in range(0, len(x)):
            w.append(1 + (x[idx] - 1) / 4)
        w = np.array(w)

        term1 = (np.sin(np.pi*w[0]))**2
        term3 = (w[-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[-1]))**2)
        term2 = 0
        for idx in range(1, len(w)):
            wi = w[idx]
            new = (wi-1)**2 * (1 + 10 * (np.sin(np.pi * wi + 1))**2)
            term2 = term2 + new

        result = term1 + term2 + term3

        if not self.minimize:
            return -result
        else:
            return result
