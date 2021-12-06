import functools
import inspect
import warnings
from typing import Type, Callable, Optional

import torch
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
            loss_, eq_defect_, ineq_defect_ = closure()
            if self.shrinkage is not None and eq_defect_:
                eq_defect_ = [self.shrinkage(e) for e in eq_defect_]

            return loss_, eq_defect_, ineq_defect_

        loss, eq_defect, ineq_defect = closure_with_shrinkage()

        if not self.eq_multipliers and not self.ineq_multipliers:
            self.init_dual_variables(eq_defect, ineq_defect, dtype=self.dual_dtype)

        assert eq_defect is None or all([validate_defect(d, m) for d, m in zip(eq_defect, self.eq_multipliers)])
        assert ineq_defect is None or all([d.shape == m.shape for d, m in zip(ineq_defect, self.ineq_multipliers)])

        lagrangian = self.minmax_backward(loss, eq_defect, ineq_defect)

        should_back_prop = False
        if hasattr(self.primal_optimizer, "extrapolation"):
            self.primal_optimizer.extrapolation(loss)
            should_back_prop = True

        # RYAN: this is not necessary
        if not self.alternating and hasattr(self.dual_optimizer, "extrapolation"):
            self.dual_optimizer.extrapolation(loss)
            should_back_prop = True

        if should_back_prop:
            loss_, eq_defect_, ineq_defect_ = closure_with_shrinkage()
            lagrangian_ = self.minmax_backward(loss_, eq_defect_, ineq_defect_)

        self.primal_optimizer.step()

        if self.alternating:
            loss_, eq_defect_, ineq_defect_ = closure_with_shrinkage()
            lagrangian_ = self.minmax_backward(loss_, eq_defect_, ineq_defect_)
        self.dual_optimizer.step()

        return lagrangian, loss, eq_defect, ineq_defect

    def minmax_backward(self, loss, eq_defect, ineq_defect):
        self.primal_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()
        rhs = self.weighted_constraint(eq_defect, ineq_defect)

        if self.augmented_lagrangian_coefficient:
            lagrangian = loss + self.augmented_lagrangian_coefficient * self.squared_sum_constraint(eq_defect, ineq_defect) + sum(rhs)
        else:
            lagrangian = loss + sum(rhs)

        lagrangian.backward()
        [m.weight.grad.mul_(-1) for m in self.eq_multipliers]
        [m.weight.grad.mul_(-1) for m in self.ineq_multipliers]
        return lagrangian.item()

    def squared_sum_constraint(self, eq_defect, ineq_defect) -> torch.Tensor:
        ''' Compute quadratic penalty for augmented Lagrangian
        '''
        if eq_defect is not None:
            constraint_sum = torch.zeros(1, device=eq_defect[0].device)
        else:
            constraint_sum = torch.zeros(1, device=ineq_defect[0].device)

        for defect in [eq_defect, ineq_defect]:
            if defect is not None:
                for hi in defect:
                    if hi.is_sparse:
                        hi = hi.coalesce().values()
                    constraint_sum += torch.sum(torch.square(hi))

        return constraint_sum

    def weighted_constraint(self, eq_defect, ineq_defect) -> list:
        """Compute contribution of the constraints, weighted by current multiplier values

        Returns:
            rhs: List of contribution per constraint to the Lagrangian
        """
        rhs = []
        
        if eq_defect is not None:
            for multiplier, hi in zip(self.eq_multipliers, eq_defect):
                rhs.append(constraint_dot(hi, multiplier))

        if ineq_defect is not None:
            for multiplier, hi in zip(self.ineq_multipliers, ineq_defect):
                rhs.append(constraint_dot(hi, multiplier))
                
        return rhs

    def init_dual_variables(self, eq_defect, ineq_defect, dtype=None):
        eq_multipliers = []
        ineq_multipliers = []

        if eq_defect is not None:
            for hi in eq_defect:
                assert hi.ndim == 2, f"2d shape (batch_size, defect_size) required, found {hi.ndim}"
                if hi.is_sparse:
                    m_i = _SparseMultiplier(hi, dtype=dtype)
                else:
                    m_i = _DenseMultiplier(hi, dtype=dtype)
                eq_multipliers.append(m_i)

        if ineq_defect is not None:
            for hi in ineq_defect:
                assert hi.ndim == 2, "shape (batch_size, *) required"
                if hi.is_sparse:
                    m_i = _SparseMultiplier(hi, dtype=dtype, positive=True)
                else:
                    m_i = _DenseMultiplier(hi, dtype=dtype, positive=True)
                ineq_multipliers.append(m_i)

        self.state["eq_multipliers"] = torch.nn.ModuleList(eq_multipliers)
        self.state["ineq_multipliers"] = torch.nn.ModuleList(ineq_multipliers)

        self.dual_optimizer = self.dual_optimizer_class([
            *self.state["eq_multipliers"].parameters(),
            *self.state["ineq_multipliers"].parameters(),
        ])
        if self.augmented_lagrangian_coefficient and (hasattr(self.primal_optimizer, "extrapolation") or hasattr(self.dual_optimizer, "extrapolation")):
            warnings.warn("not sure if there is need to mix extrapolation and augmented lagrangian")

    def eval_multipliers(self, mult_type='ineq'):            
        return [_.forward().item() for _ in self.state[mult_type + "_multipliers"]]
    
    @property
    def ineq_multipliers(self):
        return self.state["ineq_multipliers"]

    @property
    def eq_multipliers(self):
        return self.state["eq_multipliers"]


def constraint_dot(defect, multiplier):
    """Compute constraint contribution for given (potent. sparse) defect and multiplier
    """
    if defect.is_sparse:
        hi = defect.coalesce()
        indices = hi.indices().squeeze(0)
        return torch.einsum('bh,bh->', multiplier(indices).to(dtype=hi.dtype), hi.values())
    else:
        return torch.sum(multiplier().to(dtype=defect.dtype) * defect) 
        
def validate_defect(defect, multiplier):
    if defect.is_sparse:
        return defect.shape == multiplier.shape
    else:
        return defect.shape == multiplier(defect).shape
