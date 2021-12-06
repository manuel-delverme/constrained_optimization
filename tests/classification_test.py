# from unittest.mock import patch

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import src.torch_constrained as tc

import torch
import matplotlib.pyplot as plt

# Create classification problem with random Gaussian inputs and random class labels
# Using more dimensions than datapoints
batch_size, num_features, num_classes = 100, 200, 3
X = torch.randn(batch_size, num_features)
Y = torch.randint(0, num_classes, (batch_size, ))

# Use a simple linear model 
model = torch.nn.Linear(num_features, num_classes, bias=False)
loss_fn = torch.nn.CrossEntropyLoss()

def closure():
    outputs = model(X)
    loss = loss_fn(model(X), Y) # Cross Entropy
    norm1 = torch.linalg.norm(model.weight, ord=1) 
    norm2 = torch.linalg.norm(model.weight, ord=2)

    eq_defects = None
    # Constrain model to have norm 1 below 0.5, and norm 2 below 1
    ineq_defects = [norm1 - 1., norm2 - 1.]
    
    # User can store model output and other 'miscellaneous' objects in misc dict
    misc = {'outputs': outputs}

    closure_dict = {'loss': loss,
                    'eq_defect': eq_defects,
                    'ineq_defect': ineq_defects,
                    'misc': misc}

    return closure_dict


optimizer = tc.ConstrainedOptimizer(
        primal_optim_class=tc.OPTIM_DICT['Adam'],
        primal_kwargs={'lr': 1e-3},
        primal_parameters=model.parameters(),
        dual_optim_class=tc.OPTIM_DICT['Adam'],
        dual_kwargs={'lr': 1e-1},
        alternating=False,
        dual_reset=True
    )


log_dict = {'loss': [], 'acc': []}

for _ in range(5000):
    step_dict = optimizer.step(closure)
    log_dict['loss'].append((_, step_dict['loss'].item()))
    
    if _ % 500 == 0: 
        print('Loss', step_dict['loss'].item())
        
        print('Eq defects:', step_dict['eq_defect'])
        print('Ineq defects:', step_dict['ineq_defect'])
        
        print('Mults', optimizer.eval_multipliers('ineq'))
        _accuracy = (step_dict['misc']['outputs'].argmax(dim=1) == Y).float().mean().item()
        log_dict['acc'].append((_, _accuracy))
        print('Accuracy', _accuracy)

plt.plot([_[0] for _ in log_dict['loss']], [_[1] for _ in log_dict['loss']])
plt.plot([_[0] for _ in log_dict['acc']], [_[1] for _ in log_dict['acc']])
plt.show()

