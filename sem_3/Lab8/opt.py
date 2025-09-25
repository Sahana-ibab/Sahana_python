import torch 
class SGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

class SGD_Momentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.v = [torch.zeros_like(p) for p in self.params]
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.v[i] = self.momentum * self.v[i] - self.lr * p.grad
                p.data += self.v[i]

class AdaGrad:
    def __init__(self, params, lr=0.01, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.eps = eps
        self.g2 = [torch.zeros_like(p) for p in self.params]
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.g2[i] += p.grad * p.grad
                p.data -= self.lr * p.grad / (torch.sqrt(self.g2[i]) + self.eps)
