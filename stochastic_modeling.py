import matplotlib.pyplot
from random import gauss


class ItoProcess:
    dt = 0.001
    t_max = 1.0
    n_max = 1000
    t = [0.0]
    while t[-1] < t_max:
        t.append(t[-1] + dt)

    def __init__(self, mu=lambda x, t: 0.0, sigma=lambda x, t: 1.0, x_0=lambda: 0.0):
        self.mu = mu
        self.sigma = sigma
        self.x_0 = x_0

    def x_next(self, x, t):
        return x + self.mu(x, t) * ItoProcess.dt + self.sigma(x, t) * gauss(0.0, ItoProcess.dt ** 0.5)

    def trajectory(self):
        t = 0.0
        x = [self.x_0()]
        while t < ItoProcess.t_max:
            x.append(self.x_next(x[-1], t))
            t += ItoProcess.dt
        return x

    def expectation(self, p=1):
        expectation = 0.0
        for n in range(ItoProcess.n_max):
            expectation += self.trajectory()[-1] ** p
        return expectation / ItoProcess.n_max

    def variance(self):
        return self.expectation(2) - self.expectation() ** 2

    def plot(self):
        matplotlib.pyplot.plot(ItoProcess.t, self.trajectory())
        matplotlib.pyplot.show()

    def asymptotics(self, asymptotics):
        t = 0.0
        x = [self.x_0()]
        while t < ItoProcess.t_max:
            x.append(self.x_next(x[-1], t))
            t += ItoProcess.dt
        return x[-1] / asymptotics(t - ItoProcess.dt)


r = ItoProcess(lambda x, t: x + 1.0 / (2.0 * x), x_0=lambda: 1.0)
r.plot()
print("E[R(" + str(ItoProcess.t_max) + ")] =", r.expectation())
print("E[R²(" + str(ItoProcess.t_max) + ")] =", r.expectation(2))
