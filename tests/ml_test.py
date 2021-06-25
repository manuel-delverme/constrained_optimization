# from unittest.mock import patch
import torch
import torch_constrained

import HockSchittkowski


# import torch_constrained

def load_HockSchittkowski_models():
    for name, model in HockSchittkowski.__dict__.items():
        if isinstance(model, type) and issubclass(model, HockSchittkowski.Hs) and model != HockSchittkowski.Hs:
            yield model.objective_function, model.constraints, model.optimal_solution, model.initialize


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

for objective_function, constraints, optimal_solution, initialize in benchmarks:
    x, = initialize()
    x.requires_grad = True
    eta = 0.05

    optimizer = torch_constrained.ConstrainedOptimizer(
        torch_constrained.ExtraAdagrad,
        torch_constrained.ExtraAdagrad,
        lr_x=eta,
        lr_y=eta,
        primal_parameters=[x, ],
    )
    for _ in range(100):
        optimizer.step(lambda: (-objective_function(x), [constraints(x).reshape(-1, 1), ], None))
        print(x.detach().numpy(), objective_function(x).item(), constraints(x).item())
    assert torch.allclose(x.item(), optimal_solution)
