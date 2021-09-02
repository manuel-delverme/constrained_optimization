import functools
import inspect
import warnings
from typing import Type, List, Callable, Optional, Union

import torch
import torch.nn

from .multipliers import _SparseMultiplier, _DenseMultiplier

class ConstrainedOptimizer(torch.optim.Optimizer):
    def __init__(
            self,
            primal_optim_class: Type[torch.optim.Optimizer],
            dual_optim_class: Type[torch.optim.Optimizer],
            lr_primal: float,
            lr_dual: float,
            primal_parameters,
            augmented_lagrangian_coefficient=False,
            alternating=False,
            shrinkage: Optional[Callable] = None,
            dual_dtype=None,
            dual_reset=False,
            verbose=False,
    ):
        
        # Instantiate user-specified primal optimizer
        if inspect.isgenerator(primal_parameters):
            primal_parameters = list(primal_parameters)
        self.primal_optimizer = primal_optim_class(primal_parameters, lr_primal)
        self.primal_parameters = primal_parameters

        # If required, the dual optimizer is instantiated in 'init_dual_variables'
        self.dual_optimizer = None
        self.dual_optimizer_class = functools.partial(dual_optim_class, lr=lr_dual)
        self.dual_dtype = dual_dtype
        
        self.dual_reset = dual_reset
        
        # Other optimization and lagrangian options
        self.augmented_lagrangian_coefficient = augmented_lagrangian_coefficient
        self.alternating = alternating
        self.shrinkage = shrinkage
        
        self.verbose = verbose 

        super().__init__(primal_parameters, {})

    def step(self, closure):
        def closure_with_shrinkage():
            closure_dict = closure()

            if self.shrinkage is not None and closure_dict['eq_defect']:
                closure_dict['eq_defect'] = [self.shrinkage(e) for e in closure_dict['eq_defect']]

            return closure_dict

        closure_dict = closure_with_shrinkage()
        loss, eq_defect, ineq_defect = [closure_dict[_] for _ in ['loss', 'eq_defect', 'ineq_defect']]

        if not self.eq_multipliers and not self.ineq_multipliers:
            # If not done before, initialize dual variables
            # This step also instantiates dual_optimizer
            self.init_dual_variables(eq_defect, ineq_defect, dtype=self.dual_dtype)

        assert eq_defect is None or all([validate_defect(d, m) for d, m in zip(eq_defect, self.eq_multipliers)])
        assert ineq_defect is None or all([d.shape == m.shape for d, m in zip(ineq_defect, self.ineq_multipliers)])

        lagrangian = self.lagrangian_backward(loss, eq_defect, ineq_defect)
        closure_dict['lagrangian'] = lagrangian
        
        # JGP TODO: Why was this being applied on the object loss? 
        # Shouldn't this be called with input lagrangian? Otherwise subsequent
        # extrapolation backprops will ignore constraints.
        self.run_optimizers_steps(lagrangian, closure_with_shrinkage)
        
        return closure_dict

    def run_optimizers_steps(self, loss, closure_fn):
        
        should_back_prop = False
        if hasattr(self.primal_optimizer, "extrapolation"):
            self.primal_optimizer.extrapolation(loss)
            should_back_prop = True

        # RYAN: this is not necessary
        if not self.alternating and hasattr(self.dual_optimizer, "extrapolation"):
            self.dual_optimizer.extrapolation(loss)
            should_back_prop = True

        if should_back_prop:
            closure_dict_ = closure_fn()
            in_tuple = (closure_dict_[_] for _ in ['loss', 'eq_defect', 'ineq_defect'])
            lagrangian_ = self.lagrangian_backward(*in_tuple)

        self.primal_optimizer.step()

        if self.alternating:
            # Once having updated primal parameters, re-compute gradient
            # Skip gradient wrt model parameters to avoid wasteful computation
            # as we only need gradient wrt multipliers.
            closure_dict_ = closure_fn()
            in_tuple = (closure_dict_[_] for _ in ['loss', 'eq_defect', 'ineq_defect'])
            lagrangian_ = self.lagrangian_backward(*in_tuple, ignore_primal=True)

        if self.dual_reset:
            # 'Reset' value of multiplier to zero as soon as solution becomes feasible 
            for multiplier in self.ineq_multipliers:
                # Call to lagrangian_backward has already flipped sign
                # Currently positive sign means original defect is negative = feasible
                if multiplier.weight.grad.item() > 0:
                    multiplier.weight.grad *= 0
                    multiplier.weight.data *= 0

        self.dual_optimizer.step()


    def init_dual_variables(self, 
                            eq_defect=Optional[List[torch.tensor]],
                            ineq_defect=Optional[List[torch.tensor]],
                            dtype=None):
        """Initialize dual variables and optimizers given list of equality and
        inequality defects.

        Args:
            eq_defect ([list of tensor], optional): Defects for equality constraints 
            ineq_defect ([list of tensor], optional): Defects for inequality constraints.
            dtype ([type], optional): Data type for multipliers.
        """

        if self.verbose:
            print('Initializing dual variables')

        aux_dict = {'eq': eq_defect,
                    'ineq': ineq_defect}

        for const_name, const_defects in aux_dict.items():

            # For each constraint type, create a multiplier for each constraint

            multipliers = []
            if const_defects is not None:
                for hi in const_defects:
                    # assert hi.ndim == 2, f"2d shape (batch_size, defect_size) required, found {hi.ndim}"
                    
                    # Force positivity if dealing with inequality
                    kwargs = {'dtype': dtype, 'positive': const_name=='ineq'}
                    m_i = _SparseMultiplier(hi, **kwargs) if hi.is_sparse else _DenseMultiplier(hi, **kwargs)

                    multipliers.append(m_i)

            # Join multipliers per constraint type into one module list
            self.state[const_name + "_multipliers"] = torch.nn.ModuleList(multipliers)
        
        # Initialize dual optimizer in charge of newly created dual parameters
        self.dual_optimizer = self.dual_optimizer_class([
            *self.state["eq_multipliers"].parameters(),
            *self.state["ineq_multipliers"].parameters(),
            ])

        # TODO: Math backing up this warning?
        using_extrapolation =  (hasattr(self.primal_optimizer, "extrapolation") or hasattr(self.dual_optimizer, "extrapolation"))
        using_augmented_lagrangian = bool(self.augmented_lagrangian_coefficient)
        if using_augmented_lagrangian and using_extrapolation:
            warnings.warn("not sure if there is need to mix extrapolation and augmented lagrangian")

    def lagrangian_backward(self, loss, eq_defect, ineq_defect, ignore_primal=False):
        """Compute Lagrangian and backward pass

        """
        self.primal_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()
        
        lagrangian = self.compute_lagrangian(loss, eq_defect, ineq_defect)

        # Compute gradients
        if ignore_primal:
            mult_params = [m.weight for m in self.eq_multipliers]
            mult_params += [m.weight for m in self.ineq_multipliers]
            lagrangian.backward(inputs=mult_params)
        else:
            lagrangian.backward()

        # Flip gradients for dual variables to perform ascent
        [m.weight.grad.mul_(-1) for m in self.eq_multipliers]
        [m.weight.grad.mul_(-1) for m in self.ineq_multipliers]

        return lagrangian.item()

    def compute_lagrangian(self, loss, eq_defect, ineq_defect, ignore_loss=False):
        
        # Compute contribution of the constraints, weighted by current multiplier values
        rhs = self.weighted_constraint(eq_defect, ineq_defect)
        
        # Lagrangian = loss + dot(multipliers, defects)
        lagrangian = loss + sum(rhs) 

        # If using augmented Lagrangian, add squared sum of constraints
        if self.augmented_lagrangian_coefficient:
            ssc = self.squared_sum_constraint(eq_defect, ineq_defect)
            lagrangian += self.augmented_lagrangian_coefficient * ssc
        
        return lagrangian

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

    @property
    def ineq_multipliers(self):
        return self.state["ineq_multipliers"]

    @property
    def eq_multipliers(self):
        return self.state["eq_multipliers"]

    def str_eq_multipliers(self):
        return [_.forward().item() for _ in self.state["eq_multipliers"]]

    def str_ineq_multipliers(self):
        return [_.forward().item() for _ in self.state["ineq_multipliers"]]

def constraint_dot(defect, multiplier):
    """Compute constraint contribution for given (potent. sparse) defect and multiplier
    """
    if defect.is_sparse:
        hi = defect.coalesce()
        indices = hi.indices().squeeze(0)
        return torch.einsum('bh,bh->', multiplier(indices).to(dtype=hi.dtype), hi.values())
    else:
        return torch.sum(multiplier().to(dtype=defect.dtype) * defect) 
        # return torch.einsum('bh,bh->', multiplier().to(dtype=dtypes), vals)
        
def validate_defect(defect, multiplier):
    if defect.is_sparse:
        return defect.shape == multiplier.shape
    else:
        return defect.shape == multiplier(defect).shape
