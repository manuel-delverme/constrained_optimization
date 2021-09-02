# from unittest.mock import patch
import torch

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import src.torch_constrained as tc

batch_size, num_features, num_classes = 100, 200, 3
X = torch.randn(batch_size, num_features)
model = torch.nn.Linear(num_features, num_classes)
Y = torch.randint(0, num_classes, (batch_size, ))
loss_fn = torch.nn.CrossEntropyLoss()

def closure():
    outputs = model(X)
    loss = loss_fn(model(X), Y)
    norm1 = torch.linalg.norm(model.weight, ord=1) 
    norm2 = torch.linalg.norm(model.weight, ord=2)

    eq_defects = None
    ineq_defects = [norm1 - 1, norm2 - 3]
    
    # User can store model output and other 'miscellaneous' objects in misc dict
    misc = {'outputs': outputs}

    closure_dict = {'loss': loss,
                    'eq_defect': eq_defects,
                    'ineq_defect': ineq_defects,
                    'misc': misc}

    return closure_dict


optimizer = tc.ConstrainedOptimizer(
        torch.optim.Adam,
        torch.optim.SGD,
        lr_primal=1e-2,
        lr_dual=1e-1,
        primal_parameters=model.parameters(),
        alternating=True
    )


for _ in range(10000):
    step_dict = optimizer.step(closure)
    if _ % 1000 == 0: 
        print(step_dict['loss'].item(), step_dict['eq_defect'], step_dict['ineq_defect'])
        print(optimizer.str_eq_multipliers(), optimizer.str_ineq_multipliers())
        print('Acc', (step_dict['misc']['outputs'].argmax(dim=1) == Y).float().mean().item())