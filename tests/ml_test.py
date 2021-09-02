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

# class TestHS(TestCase):
#     # @patch.object(constants, "filepath_data", "./data/smol.csv")
#     def test_train(self):
#         objective_function, constraints, optimal_solution, initialize = benchmarks
#         x0 = initialize()
#         x = torch.tensor([[x0, ], ], requires_grad=True)
#         optimizer = torch_constrained.ConstrainedOptimizer(
#             torch.optim.SGD,
#             torch.optim.SGD,
#             lr_x=eta,
#             lr_y=eta,
#             primal_parameters=[x, ],
#         )
#
#         optimizer = torch_constrained.ConstrainedOptimizer(
#             torch_constrained.ExtraAdagrad,
#             torch_constrained.ExtraSGD,
#             lr_x=1.,
#             lr_y=1.,
#             primal_parameters=list(primal.parameters()),
#         )
#         _ = train.train()

for name, model, objective_function, constraints, optimal_solution, initialize in benchmarks:
    x, = initialize()
    objective_fn = lambda u: - objective_function(u)
    x.requires_grad = True
    eta = 0.005

    optimizer = tc.ConstrainedOptimizer(
        torch.optim.Adam,
        torch.optim.Adam,
        # tc.ExtraAdagrad,
        # tc.ExtraAdagrad,
        lr_primal=eta,
        lr_dual=eta,
        primal_parameters=[x, ],
    )

    def closure():
        return {'loss': objective_fn(x),
                'eq_defect': model.constraints(x),
                'ineq_defect': None,
                'misc': None}

    print('Running', name)
    for _ in range(2000):
        optimizer.step(closure)
        if _ % 500 == 0:
            print(name, [_(x).data.numpy() for _ in [lambda u: x, objective_fn, constraints]])

    # assert torch.allclose(x, optimal_solution.float(), rtol=1e-2)
