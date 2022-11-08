from tensor import Tensor
import functional as F


class Optimizer():
    def __init__(self, params):
        self.parameters = params
    
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

class SGD(Optimizer):
    def __init__(self, params, lr, momentum=0, dampening=0, decay=0, maximize=False):
        super().__init__(params)
        self.lr=lr
        self.momentum=momentum
        self.dampening=dampening
        self.decay=decay
        self.maximize=maximize
        self.prev_b = None
    
    def step(self):
        for p in self.parameters:
            g = p.grad
            if self.decay != 0:
                g += self.decay*p.value
            
            if self.momentum != 0:
                if self.prev_b is None:
                    b = g
                else:
                    b = self.momentum*self.prev_b + (1-self.dampening)*g
                self.prev_b = b
            
            if self.maximize:
                p.value += self.lr*g
            else:
                p.value -= self.lr*g

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), decay=0, maximize=False, eps=1e-8):
        super().__init__(params)
        self.lr=lr
        self.betas=betas
        self.decay=decay
        self.maximize = maximize
        self.eps = eps
        self.prev_m = 0
        self.prev_v = 0
        self.t = 1
    
    def step(self):
        for p in self.parameters:
            g = p.grad
            if self.maximize:
                g = -1*g
            
            if self.decay != 0:
                g = g + self.decay*p.value
            
            m = self.betas[0]*self.prev_m + (1-self.betas[0]) * g
            v = self.betas[1]*self.prev_v + (1-self.betas[1]) * g**2
            self.prev_m = m
            self.prev_v = v
            m /= (1-self.betas[0]**self.t)
            v /= (1-self.betas[1]**self.t)

            p.value -= self.lr*m/(v**0.5 + self.eps)
        self.t += 1