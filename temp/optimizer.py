import torch

def get_optimizer(parameters, lr=1e-4, optimizer_name='Adam', **kwargs):
    if optimizer_name in ['Adam',]:
        optimizer = BaseOptimizer(parameters, lr=lr, optimizer_name=optimizer_name, **kwargs)
    else:
        raise NotImplementedError
    return optimizer

class BaseOptimizer(object):
    def __init__(self, parameters, lr=1e-4, optimizer_name='Adam', **kwargs):
        if optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(params=parameters, lr=lr, **kwargs)

    def step(self, closure=None):
        self.optimizer.step(closure)
    def zero_grad(self):
        self.optimizer.zero_grad()
        
