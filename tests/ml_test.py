# from unittest.mock import patch
import torch

import HockSchittkowski

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import src.torch_constrained as tc

def load_HockSchittkowski_models():
    for name, model in HockSchittkowski.__dict__.items():
        if isinstance(model, type) and issubclass(model, HockSchittkowski.Hs) and model != HockSchittkowski.Hs:
            yield name, model, model.objective_function, model.constraints, model.optimal_solution, model.initialize


benchmarks = list(load_HockSchittkowski_models())


for name, model, objective_function, constraints, optimal_solution, initialize in benchmarks:
    x, = initialize()
    objective_fn = lambda u: - objective_function(u)
    x.requires_grad = True
    eta = 0.05

    optimizer = tc.ConstrainedOptimizer(
        primal_optim_class=tc.OPTIM_DICT['ExtraAdagrad'],
        lr_primal=eta,
        primal_parameters=[x, ],
        dual_optim_class=tc.OPTIM_DICT['ExtraAdagrad'],
        lr_dual=eta,
    )

    def closure():
        return {'loss': -objective_fn(x),
                'eq_defect': model.constraints(x),
                'ineq_defect': None,
                'misc': None}

    print('\nRunning', name)
    for _ in range(5000):
        optimizer.step(closure)
        if _ % 1000 == 0:
            print(name, [str(_(x).data.numpy().round(5)) for _ in [lambda u: x, objective_fn, constraints]])

    
    found_obj = -objective_fn(x)
    assert torch.abs(found_obj - optimal_solution.float()) < 1e-2, \
        print(found_obj, optimal_solution.float())