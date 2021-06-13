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
        loss, eq_defect, _ineq_defect = closure()
        assert _ineq_defect is None, NotImplementedError

        if self.multipliers is None:
            self.init_dual_variables(eq_defect)

        assert all([d.shape == m.shape for d, m in zip(eq_defect, self.multipliers)])

        self.primal_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()

        (loss + sum(self.weighted_constraint(eq_defect))).backward()
        [m.weight.grad.mul_(-1) for m in self.multipliers]

        if self.extrapolation:
            self.primal_optimizer.extrapolation()
            # RYAN: this is not necessary
            self.dual_optimizer.extrapolation()

            self.primal_optimizer.zero_grad()
            self.dual_optimizer.zero_grad()
            loss_, eq_defect_, _ = closure()

            (loss_ + sum(self.weighted_constraint(eq_defect_))).backward()
            [m.weight.grad.mul_(-1) for m in self.multipliers]

        self.primal_optimizer.step()
        self.dual_optimizer.step()

        return loss, eq_defect

    def weighted_constraint(self, eq_defect) -> list:
        rhs = []
        for multiplier, hi in zip(self.state["multipliers"], eq_defect):
            if hi.is_sparse:
                hi = hi.coalesce()
                indices = hi.indices().squeeze(0)
                rhs.append(torch.einsum('bh,bh->', multiplier(indices), hi.values()))
            else:
                rhs.append(torch.einsum('bh,bh->', multiplier, hi))
        return rhs

    def init_dual_variables(self, h):
        multipliers = []
        for hi in h:
            if hi.is_sparse:
                m_i = SparseMultiplier(hi)
            else:
                m_i = DenseMultiplier(hi)
            multipliers.append(m_i)

        self.multipliers = torch.nn.ModuleList(multipliers)
        self.state["multipliers"] = self.multipliers
        self.dual_optimizer = self.dual_optimizer_class(self.multipliers.parameters())


class SparseMultiplier(torch.nn.Embedding):
    def __init__(self, hi):
        super().__init__(*hi.shape, _weight=torch.zeros(hi.shape, device=hi.device), sparse=True)

    @property
    def shape(self):
        return self.weight.shape


class DenseMultiplier(torch.nn.Module):
    def __init__(self, hi):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(hi.shape, device=hi.device))

    @property
    def shape(self):
        return self.weight.shape

    @property
    def forward(self):
        return self.weight
