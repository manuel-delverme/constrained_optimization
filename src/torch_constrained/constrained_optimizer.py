import functools
import inspect
import warnings
from typing import Type, Callable, Optional

import torch
import torch.nn
from .multipliers import _SparseMultiplier, _DenseMultiplier


class ConstrainedOptimizer(torch.optim.Optimizer):
    def __init__(
            self,
            loss_optimizer: Type[torch.optim.Optimizer],
            constraint_optimizer: Type[torch.optim.Optimizer],
            lr_x,
            lr_y,
            primal_parameters,
            augmented_lagrangian_coefficient=False,
            alternating=False,
            shrinkage: Optional[Callable] = None,
            dual_dtype=None,
            primal_kwargs={},
            dual_kwargs={},
    ):
        if inspect.isgenerator(primal_parameters):
            primal_parameters = list(primal_parameters)

        self.primal_optimizer = loss_optimizer(primal_parameters, lr_x, **primal_kwargs)
        self.dual_optimizer_class = functools.partial(constraint_optimizer, lr=lr_y, **dual_kwargs)

        self.dual_optimizer = None
        self.augmented_lagrangian_coefficient = augmented_lagrangian_coefficient
        self.alternating = alternating
        self.shrinkage = shrinkage
        self.dual_dtype = dual_dtype

        super().__init__(primal_parameters, {})

    def step(self, closure):
        def closure_with_shrinkage():
            loss_, eq_defect_, inequality_defect_ = closure()
            if self.shrinkage is not None and eq_defect_:
                eq_defect_ = [self.shrinkage(e) for e in eq_defect_]

            return loss_, eq_defect_, inequality_defect_

        loss, eq_defect, inequality_defect = closure_with_shrinkage()

        if not self.equality_multipliers and not self.inequality_multipliers:
            self.init_dual_variables(eq_defect, inequality_defect, dtype=self.dual_dtype)

        assert eq_defect is None or all([validate_defect(d, m) for d, m in zip(eq_defect, self.equality_multipliers)])
        assert inequality_defect is None or all([d.shape == m.shape for d, m in zip(inequality_defect, self.inequality_multipliers)])

        lagrangian = self.minmax_backward(loss, eq_defect, inequality_defect)

        should_back_prop = False
        if hasattr(self.primal_optimizer, "extrapolation"):
            self.primal_optimizer.extrapolation(loss)
            should_back_prop = True

        # RYAN: this is not necessary
        if not self.alternating and hasattr(self.dual_optimizer, "extrapolation"):
            self.dual_optimizer.extrapolation(loss)
            should_back_prop = True

        if should_back_prop:
            loss_, eq_defect_, inequality_defect_ = closure_with_shrinkage()
            lagrangian_ = self.minmax_backward(loss_, eq_defect_, inequality_defect_)

        self.primal_optimizer.step()

        if self.alternating:
            loss_, eq_defect_, inequality_defect_ = closure_with_shrinkage()
            lagrangian_ = self.minmax_backward(loss_, eq_defect_, inequality_defect_)
        self.dual_optimizer.step()

        return lagrangian, loss, eq_defect, inequality_defect

    def minmax_backward(self, loss, eq_defect, inequality_defect):
        self.primal_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()
        rhs = self.weighted_constraint(eq_defect, inequality_defect)

        if self.augmented_lagrangian_coefficient:
            lagrangian = loss + self.augmented_lagrangian_coefficient * self.squared_sum_constraint(eq_defect, inequality_defect) + sum(rhs)
        else:
            lagrangian = loss + sum(rhs)

        lagrangian.backward()
        [m.weight.grad.mul_(-1) for m in self.equality_multipliers]
        [m.weight.grad.mul_(-1) for m in self.inequality_multipliers]
        return lagrangian.item()

    def squared_sum_constraint(self, eq_defect, inequality_defect) -> torch.Tensor:
        if eq_defect is not None:
            constraint_sum = torch.zeros(1, device=eq_defect[0].device)
        else:
            constraint_sum = torch.zeros(1, device=inequality_defect[0].device)

        if eq_defect is not None:
            for hi in eq_defect:
                if hi.is_sparse:
                    hi = hi.coalesce().values()
                constraint_sum += torch.sum(torch.square(hi))

        if inequality_defect is not None:
            for hi in inequality_defect:
                if hi.is_sparse:
                    hi = hi.coalesce().values()
                constraint_sum += torch.sum(torch.square(hi))
        return constraint_sum

    def weighted_constraint(self, eq_defect, inequality_defect) -> list:
        rhs = []

        if eq_defect is not None:
            for multiplier, hi in zip(self.equality_multipliers, eq_defect):
                if hi.is_sparse:
                    hi = hi.coalesce()
                    indices = hi.indices().squeeze(0)
                    rhs.append(torch.einsum('bh,bh->', multiplier(indices).to(dtype=hi.dtype), hi.values()))
                else:
                    rhs.append(torch.einsum('bh,bh->', multiplier(hi).to(dtype=hi.dtype), hi))

        if inequality_defect is not None:
            for multiplier, hi in zip(self.inequality_multipliers, inequality_defect):
                if hi.is_sparse:
                    hi = hi.coalesce()
                    indices = hi.indices().squeeze(0)
                    rhs.append(torch.einsum('bh,bh->', multiplier(indices).to(dtype=hi.dtype), hi.values()))
                else:
                    rhs.append(torch.einsum('bh,bh->', multiplier(hi).to(dtype=hi.dtype), hi))
        return rhs

    def init_dual_variables(self, equality_defect, inequality_defect, dtype=None):
        equality_multipliers = []
        inequality_multipliers = []

        if equality_defect is not None:
            for hi in equality_defect:
                assert hi.ndim == 2, f"2d shape (batch_size, defect_size) required, found {hi.ndim}"
                if hi.is_sparse:
                    m_i = _SparseMultiplier(hi, dtype=dtype)
                else:
                    m_i = _DenseMultiplier(hi, dtype=dtype)
                equality_multipliers.append(m_i)

        if inequality_defect is not None:
            for hi in inequality_defect:
                assert hi.ndim == 2, "shape (batch_size, *) required"
                if hi.is_sparse:
                    m_i = _SparseMultiplier(hi, dtype=dtype, positive=True)
                else:
                    m_i = _DenseMultiplier(hi, dtype=dtype, positive=True)
                inequality_multipliers.append(m_i)

        self.state["equality_multipliers"] = torch.nn.ModuleList(equality_multipliers)
        self.state["inequality_multipliers"] = torch.nn.ModuleList(inequality_multipliers)

        self.dual_optimizer = self.dual_optimizer_class([
            *self.state["equality_multipliers"].parameters(),
            *self.state["inequality_multipliers"].parameters(),
        ])
        if self.augmented_lagrangian_coefficient and (hasattr(self.primal_optimizer, "extrapolation") or hasattr(self.dual_optimizer, "extrapolation")):
            warnings.warn("not sure if there is need to mix extrapolation and augmented lagrangian")

    @property
    def inequality_multipliers(self):
        return self.state["inequality_multipliers"]

    @property
    def equality_multipliers(self):
        return self.state["equality_multipliers"]


def validate_defect(defect, multiplier):
    if defect.is_sparse:
        return defect.shape == multiplier.shape
    else:
        return defect.shape == multiplier(defect).shape
