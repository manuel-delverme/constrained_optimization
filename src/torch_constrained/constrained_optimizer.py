import inspect
import warnings
from functools import partial
from typing import Type, List, Callable, Optional, Dict

import torch
from .multipliers import _SparseMultiplier, _DenseMultiplier

class ConstrainedOptimizer(torch.optim.Optimizer):
    def __init__(
            self,
            primal_optim_class: Type[torch.optim.Optimizer],
            primal_kwargs: Dict,
            primal_parameters,
            dual_optim_class: Optional[Type[torch.optim.Optimizer]]=None,
            dual_kwargs: Optional[Dict]=None,
            ineq_init: Optional[List[torch.Tensor]]=None,
            eq_init: Optional[List[torch.Tensor]]=None,
            augmented_lagrangian_coeff=False,
            alternating=False,
            shrinkage: Optional[Callable]=None,
            dual_reset=False,
            verbose=False,
        ):
        
        # Instantiate user-specified primal optimizer
        if inspect.isgenerator(primal_parameters):
            primal_parameters = list(primal_parameters)
        self.primal_optimizer = primal_optim_class(primal_parameters, **primal_kwargs)
        self.primal_parameters = primal_parameters

        # Initialization values for the dual variables
        self.dual_init_done = False
        self.ineq_init = ineq_init
        self.eq_init = eq_init

        # Create flag for execution in un-constrained setting
        self.is_constrained = dual_optim_class is not None 

        if self.is_constrained:
            # Important: the dual optimizer is instantiated in 'init_dual_variables'.
            self.dual_optimizer_class = partial(dual_optim_class, **dual_kwargs)
            self.dual_reset = dual_reset

            # Other optimization and lagrangian options
            self.augmented_lagrangian_coeff = augmented_lagrangian_coeff
            self.alternating = alternating
            self.shrinkage = shrinkage
        else:
            print('Unconstrained Execution')
            if len(dual_kwargs) != 0:
                warnings.warn("""Did not provide dual optimizer dual kwargs is non empty.
                              Running with no dual optimizer and ignoring hyper-parameters.""")
            self.dual_optimizer_class = None
            self.dual_reset = False

            # Other optimization and lagrangian options
            self.augmented_lagrangian_coeff = 0.
            self.alternating = False
            self.shrinkage = 0.
        
        self.verbose = verbose 

        super().__init__(primal_parameters, {})

    def step(self, closure):
        def closure_with_shrinkage():
            closure_dict = closure()

            # JGP TODO: Deactivated this. What is this supposed to do?
            # if self.shrinkage is not None and closure_dict['eq_defect']:
            #     closure_dict['eq_defect'] = [self.shrinkage(e) for e in closure_dict['eq_defect']]

            return closure_dict

        closure_dict = closure_with_shrinkage()
        loss, eq_defect, ineq_defect = [closure_dict[_] for _ in ['loss', 'eq_defect', 'ineq_defect']]

        if not self.dual_init_done and not self.eq_multipliers and not self.ineq_multipliers:
            # If not done before, instantiate and initialize dual variables
            # This step also instantiates dual_optimizer, if necessary
            self.init_dual_variables(eq_defect, ineq_defect)

            # Ensure multiplier shapes match those of the defects
            assert eq_defect is None or all([validate_defect(d, m) for d, m in zip(eq_defect, self.eq_multipliers)])
            assert ineq_defect is None or all([d.shape == m.shape for d, m in zip(ineq_defect, self.ineq_multipliers)])

        # Compute lagrangian value based on current loss and values of multipliers
        lagrangian = self.lagrangian_backward(loss, eq_defect, ineq_defect)
        closure_dict['lagrangian'] = lagrangian
        
        # JGP TODO: Why was this being applied on the object loss? 
        # Shouldn't this be called with input lagrangian? Otherwise subsequent
        # extrapolation backprops will ignore constraints.
        self.run_optimizers_step(lagrangian, closure_with_shrinkage)
        
        return closure_dict

    def run_optimizers_step(self, loss, closure_fn):
        
        should_back_prop = False
        if hasattr(self.primal_optimizer, "extrapolation"):
            self.primal_optimizer.extrapolation(loss)
            should_back_prop = True

        if self.is_constrained and not self.alternating and hasattr(self.dual_optimizer, "extrapolation"):
            self.dual_optimizer.extrapolation(loss)
            should_back_prop = True

        if should_back_prop:
            closure_dict_ = closure_fn()
            in_tuple = (closure_dict_[_] for _ in ['loss', 'eq_defect', 'ineq_defect'])
            lagrangian_ = self.lagrangian_backward(*in_tuple)

        self.primal_optimizer.step()

        if self.is_constrained and self.alternating:
            # Once having updated primal parameters, re-compute gradient
            # Skip gradient wrt model parameters to avoid wasteful computation
            # as we only need gradient wrt multipliers.
            closure_dict_ = closure_fn()
            in_tuple = (closure_dict_[_] for _ in ['loss', 'eq_defect', 'ineq_defect'])
            lagrangian_ = self.lagrangian_backward(*in_tuple, ignore_primal=True)
        
        if self.dual_reset:
            # 'Reset' value of inequality multipliers to zero as soon as solution becomes feasible 
            for multiplier in self.ineq_multipliers:
                # Call to lagrangian_backward has already flipped sign
                # Currently positive sign means original defect is negative = feasible

                if multiplier.weight.grad.item() > 0:
                    multiplier.weight.grad *= 0
                    multiplier.weight.data *= 0

        if self.is_constrained:
            self.dual_optimizer.step()

    def lagrangian_backward(self, loss, eq_defect, ineq_defect, ignore_primal=False):
        """Compute Lagrangian and backward pass

        """
        self.primal_optimizer.zero_grad()

        if self.is_constrained: self.dual_optimizer.zero_grad()
        
        lagrangian = self.compute_lagrangian(loss, eq_defect, ineq_defect)

        # Compute gradients
        if ignore_primal and self.is_constrained:
                mult_params = [m.weight for m in self.eq_multipliers]
                mult_params += [m.weight for m in self.ineq_multipliers]
                lagrangian.backward(inputs=mult_params)
        else:
            lagrangian.backward()

        # Flip gradients for dual variables to perform ascent
        if self.is_constrained:
            [m.weight.grad.mul_(-1) for m in self.eq_multipliers]
            [m.weight.grad.mul_(-1) for m in self.ineq_multipliers]

        return lagrangian.item()

    def compute_lagrangian(self, loss, eq_defect, ineq_defect):
        
        # Compute contribution of the constraints, weighted by current multiplier values
        rhs = self.weighted_constraint(eq_defect, ineq_defect)
        
        # Lagrangian = loss + dot(multipliers, defects)
        lagrangian = loss + sum(rhs) 

        # If using augmented Lagrangian, add squared sum of constraints
        if self.augmented_lagrangian_coeff > 0:
            ssc = self.squared_sum_constraint(eq_defect, ineq_defect)
            lagrangian += self.augmented_lagrangian_coeff * ssc
        
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

    def init_dual_variables(self, 
                            eq_defect: Optional[List[torch.Tensor]],
                            ineq_defect: Optional[List[torch.Tensor]]
                            ):
        """Initialize dual variables and optimizers given list of equality and
        inequality defects.

        Args:
            eq_defect: Defects for equality constraints 
            ineq_defect: Defects for inequality constraints.
        """

        if self.verbose: print('Initializing dual variables')

        aux_dict = {'eq': eq_defect, 'ineq': ineq_defect}
        aux_init = {'eq': self.eq_init, 'ineq': self.ineq_init}

        for const_name, const_defects in aux_dict.items():

            multipliers = []
            if const_defects is not None:
                
                # Assert provided inits match number of constraints 
                assert aux_init[const_name] is None or len(aux_init[const_name]) == len(const_defects)
                
                # For each constraint type, create a multiplier for each constraint
                for i, defect in enumerate(const_defects):
                    # Set user-specified init values, else default to zeros
                    if aux_init[const_name] is None:
                        init_val = torch.zeros_like(defect)
                    else:
                        init_val = torch.tensor(aux_init[const_name][i], device=defect.device)
                    
                    mult_class = _SparseMultiplier if defect.is_sparse else _DenseMultiplier
                    # Force positivity if dealing with inequality
                    mult_i = mult_class(init_val, positive=const_name == "ineq")
                    multipliers.append(mult_i)

            # Join multipliers per constraint type into one module list
            self.state[const_name + "_multipliers"] = torch.nn.ModuleList(multipliers)
        
        if self.is_constrained:
            # Initialize dual optimizer in charge of newly created dual parameters
            self.dual_optimizer = self.dual_optimizer_class([
                *self.state["eq_multipliers"].parameters(),
                *self.state["ineq_multipliers"].parameters(),
                ])
        else:
            self.dual_optimizer = None

        # Mark dual instantiation an init as complete
        self.dual_init_done = True

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
    return defect.shape == multiplier.shape