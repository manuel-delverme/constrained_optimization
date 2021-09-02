import numpy as np
import torch
from torch import tensor, zeros


# Downloaded from https://apmonitor.com/wiki/uploads/Apps/hs.zip

class Hs:

    def constraints(x):
        return torch.zeros(1)

class Hs01(Hs):
    initialize = lambda: (
       torch.zeros(2),  # x
    )

    objective_function = lambda x: -(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)
    optimal_solution = tensor([1, 1])


class Hs06(Hs):
    initialize = lambda: (
       torch.zeros(2),  # x
    )

    objective_function = lambda x: -((1 - x[0]) ** 2)
    optimal_solution = tensor([1, 1])

    def constraints(x):
        h0 = lambda x: 10 * (x[1] - x[0] ** 2) - 0
        return torch.stack((h0(x), ))


class Hs07(Hs):
    initialize = lambda: (
       torch.zeros(2),  # x
    )

    objective_function = lambda x: -(torch.log(1 + x[0] ** 2) - x[1])
    optimal_solution = -tensor(-np.sqrt(3))

    def constraints(x):
        h0 = lambda x: (1 + x[0] ** 2) ** 2 + x[1] ** 2
        return torch.stack((h0(x), ))


class Hs08(Hs):
    initialize = lambda: (
       torch.zeros(2),  # x
    )

    objective_function = lambda x: -torch.tensor(-1.)
    optimal_solution = -tensor(-1.)

    def constraints(x):
        h0 = lambda x: x[0] ** 2 + x[1] ** 2
        h1 = lambda x: x[0] * x[1]
        return torch.stack((h0(x), h1(x)))


class Hs09(Hs):
    initialize = lambda: (
       torch.zeros(2),  # x
    )

    objective_function = lambda x: -(torch.sin(np.pi * x[0] / 12) *torch.cos(np.pi * x[1] / 16))
    optimal_solution = -tensor(-0.5)

    def constraints(x):
        h0 = lambda x: 4 * x[0] - 3 * x[1] - 0
        return torch.stack((h0(x), ))


class Hs26(Hs):
    initialize = lambda: (
       torch.zeros(3),  # x
    )

    objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 4)
    optimal_solution = -tensor(0)

    def constraints(x):
        h0 = lambda x: (1 + x[1] ** 2) * x[0] + x[2] ** 4
        return torch.stack((h0(x), ))


class Hs27(Hs):
    initialize = lambda: (
       torch.zeros(3),  # x
    )

    objective_function = lambda x: -((x[0] - 1) ** 2 / 100 + (x[1] - x[0] ** 2) ** 2)
    optimal_solution = -tensor(0.04)

    def constraints(x):
        h0 = lambda x: x[0] + x[2] ** 2
        return torch.stack((h0(x), ))


class Hs28(Hs):
    initialize = lambda: (
       torch.zeros(3),  # x
    )

    objective_function = lambda x: -((x[0] + x[1]) ** 2 + (x[1] + x[2]) ** 2)
    optimal_solution = -tensor(0)

    def constraints(x):
        h0 = lambda x: x[0] + 2 * x[1] + 3 * x[2]
        return torch.stack((h0(x), ))


class Hs39(Hs):
    initialize = lambda: (
       torch.zeros(4),  # x
    )

    objective_function = lambda x: -(-x[0])
    optimal_solution = -tensor(-1)

    def constraints(x):
        h0 = lambda x: x[1] - x[0] ** 3 - x[2] ** 2 - 0
        h1 = lambda x: x[0] ** 2 - x[1] - x[3] ** 2 - 0
        return torch.stack((h0(x), h1(x)))


class Hs40(Hs):
    initialize = lambda: (
       torch.zeros(4),  # x
    )

    objective_function = lambda x: -torch.tensor(-x[0] * x[1] * x[2] * x[3])
    optimal_solution = -tensor(-0.25)

    def constraints(x):
        h0 = lambda x: x[0] ** 3 + x[1] ** 2
        h1 = lambda x: x[0] ** 2 * x[3] - x[2] - 0.
        h2 = lambda x: x[3] ** 2 - x[1] - 0.
        return torch.stack((h0(x), h1(x), h2(x)))


class Hs46(Hs):
    initialize = lambda: (
       torch.zeros(5),  # x
    )

    objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[2] - 1) ** 2 + (x[3] - 1) ** 4 + (x[4] - 1) ** 6)
    optimal_solution = -tensor(0)

    def constraints(x):
        h0 = lambda x: x[0] ** 2 * x[3] + torch.sin(x[3] - x[4])
        h1 = lambda x: x[1] + x[2] ** 4 * x[3] ** 2
        return torch.stack((h0(x), h1(x)))


class Hs47(Hs):
    initialize = lambda: (
       torch.zeros(5),  # x
    )

    objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 3 + (x[2] - x[3]) ** 4 + (x[3] - x[4]) ** 4)
    optimal_solution = -tensor(0)

    def constraints(x):
        h0 = lambda x: x[0] + x[1] ** 2 + x[2] ** 3
        h1 = lambda x: x[1] - x[2] ** 2 + x[3]
        h2 = lambda x: x[0] * x[4]
        return torch.stack((h0(x), h1(x), h2(x)))


class Hs49(Hs):
    initialize = lambda: (
       torch.zeros(5),  # x
    )

    objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[2] - 1) ** 2 + (x[3] - 1) ** 4 + (x[4] - 1) ** 6)
    optimal_solution = -tensor(0)

    def constraints(x):
        h0 = lambda x: x[0] + x[1] + x[2] + x[3] + x[4] + 3 * x[3]
        h1 = lambda x: x[2] + 5 * x[4]
        return torch.stack((h0(x), h1(x)))


class Hs50(Hs):
    initialize = lambda: (
       torch.zeros(5),  # x
    )

    objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 2 + (x[2] - x[3]) ** 4 + (x[3] - x[4]) ** 2)
    optimal_solution = -tensor(0)

    def constraints(x):
        h0 = lambda x: x[0] + 2 * x[1] + 3 * x[2]
        h1 = lambda x: x[1] + 2 * x[2] + 3 * x[3]
        h2 = lambda x: x[2] + 2 * x[3] + 3 * x[4]
        return torch.stack((h0(x), h1(x), h2(x)))


class Hs51(Hs):
    initialize = lambda: (
       torch.zeros(5),  # x
    )

    objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[1] + x[2] - 2) ** 2 + (x[3] - 1) ** 2 + (x[4] - 1) ** 2)
    optimal_solution = -tensor(0)

    def constraints(x):
        h0 = lambda x: x[0] + 3 * x[1]
        h1 = lambda x: x[2] + x[3] - 2 * x[4] - 0
        h2 = lambda x: x[1] - x[4] - 0
        return torch.stack((h0(x), h1(x), h2(x)))


class Hs52(Hs):
    initialize = lambda: (
       torch.zeros(5),  # x
    )

    objective_function = lambda x: -((4 * x[0] - x[1]) ** 2 + (x[1] + x[2] - 2) ** 2 + (x[3] - 1) ** 2 + (x[4] - 1) ** 2)
    optimal_solution = -tensor(1859 / 349)

    def constraints(x):
        h0 = lambda x: x[0] + 3 * x[1] - 0
        h1 = lambda x: x[2] + x[3] - 2 * x[4] - 0
        h2 = lambda x: x[1] - x[4] - 0
        return torch.stack((h0(x), h1(x), h2(x)))


class Hs61(Hs):
    initialize = lambda: (
       torch.zeros(3),  # x
    )

    objective_function = lambda x: -(4 * x[0] ** 2 + 2 * x[1] ** 2 + 2 * x[2] ** 2 - 33 * x[0] + 16 * x[1] - 24 * x[2])
    optimal_solution = -tensor(- 143.6461422)

    def constraints(x):
        h0 = lambda x: 3 * x[0] - 2 * x[1] ** 2
        h1 = lambda x: 4 * x[0] - x[2] ** 2
        return torch.stack((h0(x), h1(x)))


class Hs77(Hs):
    initialize = lambda: (
       torch.zeros(5),  # x
    )

    objective_function = lambda x: -((x[0] - 1) ** 2 + (x[0] - x[1]) ** 2 + (x[2] - 1) ** 2 + (x[3] - 1) ** 4 + (x[4] - 1) ** 6)
    optimal_solution = -tensor(0.24150513)

    def constraints(x):
        h0 = lambda x: x[0] ** 2 * x[3] + torch.sin(x[3] - x[4])
        h1 = lambda x: x[1] + x[2] ** 4 * x[3] ** 2
        return torch.stack((h0(x), h1(x)))


class Hs78(Hs):
    initialize = lambda: (
       torch.zeros(5),  # x
    )

    objective_function = lambda x: -(x[0] * x[1] * x[2] * x[3] * x[4])
    optimal_solution = -tensor(-2.91970041)

    def constraints(x):
        h0 = lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2
        h1 = lambda x: x[1] * x[2] - 5 * x[3] * x[4] - 0
        h2 = lambda x: x[0] ** 3 + x[1] ** 3
        return torch.stack((h0(x), h1(x), h2(x)))


class Hs79(Hs):
    initialize = lambda: (
       torch.zeros(5),  # x
    )

    objective_function = lambda x: -((x[0] - 1) ** 2 + (x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 2 + (x[2] - x[3]) ** 4 + (x[3] - x[4]) ** 4)
    optimal_solution = -tensor(0.0787768209)

    def constraints(x):
        h0 = lambda x: x[0] + x[1] ** 2 + x[2] ** 3
        h1 = lambda x: x[1] - x[2] ** 2 + x[3]
        h2 = lambda x: x[0] * x[4]
        return torch.stack((h0(x), h1(x), h2(x)))
