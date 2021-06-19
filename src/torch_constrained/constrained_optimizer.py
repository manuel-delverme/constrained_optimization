import functools
from typing import Type

import torch
import torch.nn


class ConstrainedOptimizer(torch.optim.Optimizer):
    def __init__(
            self,
            loss_optimizer: Type[torch.optim.Optimizer],
            constraint_optimizer: Type[torch.optim.Optimizer],
            lr_x,
            lr_y,
            primal_parameters,
            extrapolation=True,
            augmented_lagrangian=False,
    ):
        assert not augmented_lagrangian

        self.primal_optimizer = loss_optimizer(primal_parameters, lr_x)
        self.dual_optimizer_class = functools.partial(constraint_optimizer, lr=lr_y)

        self.extrapolation = extrapolation

        self.dual_optimizer = None
        super().__init__(primal_parameters, {})

    def step(self, closure):
        loss, eq_defect, inequality_defect = closure()

        if not self.equality_multipliers and not self.equality_multipliers:
            self.init_dual_variables(eq_defect, inequality_defect)

        assert eq_defect is None or all([validate_defect(d, m) for d, m in zip(eq_defect, self.equality_multipliers)])
        assert inequality_defect is None or all([d.shape == m.shape for d, m in zip(inequality_defect, self.inequality_multipliers)])

        self.backward(loss, eq_defect, inequality_defect)

        if self.extrapolation:
            self.primal_optimizer.extrapolation()

            # RYAN: this is not necessary
            self.dual_optimizer.extrapolation()

            loss_, eq_defect_, inequality_defect_ = closure()

            self.backward(loss_, eq_defect_, inequality_defect_)

        self.primal_optimizer.step()
        self.dual_optimizer.step()

        return loss, eq_defect

    def backward(self, loss, eq_defect, inequality_defect):
        self.primal_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()
        loss.backward()
        sum(self.weighted_constraint(eq_defect, inequality_defect)).backward()
        [m.weight.grad.mul_(-1) for m in self.equality_multipliers]
        [m.weight.grad.mul_(-1) for m in self.inequality_multipliers]

    def weighted_constraint(self, eq_defect, inequality_defect) -> list:
        rhs = []

        if eq_defect is not None:
            for multiplier, hi in zip(self.equality_multipliers, eq_defect):
                if hi.is_sparse:
                    hi = hi.coalesce()
                    indices = hi.indices().squeeze(0)
                    rhs.append(torch.einsum('bh,bh->', multiplier(indices), hi.values()))
                else:
                    rhs.append(torch.einsum('bh,bh->', multiplier(hi), hi))

        if inequality_defect is not None:
            for multiplier, hi in zip(self.inequality_multipliers, inequality_defect):
                if hi.is_sparse:
                    hi = hi.coalesce()
                    indices = hi.indices().squeeze(0)
                    rhs.append(torch.einsum('bh,bh->', multiplier(indices), hi.values()))
                else:
                    rhs.append(torch.einsum('bh,bh->', multiplier(hi), hi))
        return rhs

    def init_dual_variables(self, equality_defect, inequality_defect):
        equality_multipliers = []
        inequality_multipliers = []

        if equality_defect is not None:
            for hi in equality_defect:
                if hi.is_sparse:
                    m_i = _SparseMultiplier(hi)
                else:
                    m_i = _DenseMultiplier(hi)
                equality_multipliers.append(m_i)

        if inequality_defect is not None:
            for hi in inequality_defect:
                if hi.is_sparse:
                    m_i = _SparseMultiplier(hi, positive=True)
                else:
                    m_i = _DenseMultiplier(hi, positive=True)
                inequality_multipliers.append(m_i)

        self.state["equality_multipliers"] = torch.nn.ModuleList(equality_multipliers)
        self.state["inequality_multipliers"] = torch.nn.ModuleList(inequality_multipliers)

        self.dual_optimizer = self.dual_optimizer_class([
            *self.state["equality_multipliers"].parameters(),
            *self.state["inequality_multipliers"].parameters(),
        ])

    @property
    def inequality_multipliers(self):
        return self.state["inequality_multipliers"]

    @property
    def equality_multipliers(self):
        return self.state["equality_multipliers"]


class _SparseMultiplier(torch.nn.Embedding):
    def __init__(self, hi, positive=False):
        super().__init__(*hi.shape, _weight=torch.zeros(hi.shape, device=hi.device), sparse=True)
        self.positive = positive

    @property
    def shape(self):
        return self.weight.shape

    def forward(self, *args, **kwargs):
        w = super().forward(*args, **kwargs)
        if self.positive:
            return torch.relu(w)
        else:
            return w


class _DenseMultiplier(torch.nn.Module):
    def __init__(self, hi, positive=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(hi.shape, device=hi.device))
        self.positive = positive

    @property
    def shape(self):
        return self.weight.shape

    def forward(self, h):
        w = self.weight.repeat(h.shape[0], 1)
        if self.positive:
            return torch.relu(w)
        else:
            return w


class ExtraSGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        self.old_iterate = []
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def extrapolation(self):
        """Performs the extrapolation step and save a copy of the current parameters for the update step.
        """
        if self.old_iterate:
            raise RuntimeError('Need to call step before calling extrapolation again.')
        for group in self.param_groups:
            for p in group['params']:
                self.old_iterate.append(p.detach().clone())

        # Move to extrapolation point
        super().step()

    @torch.no_grad()
    def step(self, closure=None):
        if len(self.old_iterate) == 0:
            raise RuntimeError('Need to call extrapolation before calling step.')

        i = -1
        for group in self.param_groups:
            for p in group['params']:
                i += 1
                normal_to_plane = -p.grad

                # Move back to the previous point
                p = self.old_iterate[i]
                p.grad = normal_to_plane
        super().step()

        # Free the old parameters
        self.old_iterate.clear()


class ExtraAdagrad(torch.optim.Adagrad):
    def __init__(self, *args, **kwargs):
        self.old_iterate = []
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def extrapolation(self):
        """Performs the extrapolation step and save a copy of the current parameters for the update step.
        """
        if self.old_iterate:
            raise RuntimeError('Need to call step before calling extrapolation again.')
        for group in self.param_groups:
            for p in group['params']:
                self.old_iterate.append(p.detach().clone())

        # Move to extrapolation point
        super().step()

    @torch.no_grad()
    def step(self, closure=None):
        if len(self.old_iterate) == 0:
            raise RuntimeError('Need to call extrapolation before calling step.')

        i = -1
        for group in self.param_groups:
            for p in group['params']:
                i += 1
                if p.grad is None:
                    normal_to_plane = None
                else:
                    normal_to_plane = -p.grad

                # Move back to the previous point
                p = self.old_iterate[i]
                p.grad = normal_to_plane
        super().step()

        # Free the old parameters
        self.old_iterate.clear()


def validate_defect(defect, multiplier):
    if defect.is_sparse:
        return defect.shape == multiplier.shape
    else:
        return defect.shape == multiplier(defect).shape
