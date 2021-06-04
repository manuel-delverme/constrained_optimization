import functools
from typing import Type

import torch


class ConstrainedOptimizer(torch.optim.Optimizer):
    def __init__(self, loss_optimizer: Type[torch.optim.Optimizer], constraint_optimizer: Type[torch.optim.Optimizer], lr_x, lr_y, primal_parameters, extrapolation=True):
        self.primal_optimizer = loss_optimizer(primal_parameters, lr_x)
        self.dual_optimizer_class = functools.partial(constraint_optimizer, lr=lr_y)

        self.extrapolation = extrapolation
        self.multipliers = None
        self.dual_optimizer = None
        super().__init__(primal_parameters, {})

    def step(self, closure):
        loss, eq_defect, ineq_defect = closure()
        assert ineq_defect is None, NotImplementedError

        if self.multipliers is None:
            self.init_dual_variables(eq_defect)

        self.primal_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()

        if self.extrapolation:
            (loss + self.weighted_constraint(eq_defect)).backward()
            [m.grad.mul_(-1) for m in self.multipliers]

            self.primal_optimizer.extrapolation()
            # RYAN: this is not necessary
            self.dual_optimizer.extrapolation()

            self.primal_optimizer.zero_grad()
            self.dual_optimizer.zero_grad()
            loss, eq_defect, ineq_defect = closure()

        (loss + self.weighted_constraint(eq_defect)).backward()
        [m.grad.mul_(-1) for m in self.multipliers]

        self.primal_optimizer.step()
        self.dual_optimizer.step()

    def weighted_constraint(self, eq_defect):
        rhs = 0.
        for multiplier, h_i in zip(self.state["multipliers"], eq_defect):
            # F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
            rhs += torch.einsum('bh,bh->', multiplier, h_i)
        return rhs

    def init_dual_variables(self, h):
        multipliers = []
        for hi in h:
            m_i = torch.nn.Parameter(torch.zeros(hi.shape, device=hi.device))
            multipliers.append(m_i)

        self.multipliers = torch.nn.ParameterList(multipliers)
        self.state["multipliers"] = self.multipliers
        self.dual_optimizer = self.dual_optimizer_class(self.multipliers.parameters())
