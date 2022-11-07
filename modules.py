from tensor import Tensor
import functional as F
from random import uniform
from utils import *

class Parameter(Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def zero_grad(self):
        self.grad = 0

class Module:
    def __init__(self):
        self._parameters = None
    
    def parameters(self):
        if self._parameters is None:
            parameters = []
            for attr_name, attr_value in self.__dict__.items():
                if isinstance(attr_value, self.__class__):
                    parameters.extend(attr_value.parameters())

            self._parameters = parameters

        return self._parameters

    def zero_grad(self):
        P = self.parameters()
        for p in P:
            p.zero_grad()

    def __call__(self, input):
        return self.forward(input)

class _Neuron(Module):
    def __init__(self, dim_in, bias=True, activation_fn = F.tanh):
        super().__init__()
        self.dim_in = dim_in
        self.weights = [Parameter(value = uniform(-(dim_in**-0.5), dim_in**-0.5)) for _ in range(dim_in)]
        self.bias = Parameter(value = uniform(-(dim_in**-0.5), dim_in**-0.5)) if bias else 0
        print(type(self.weights))
        self.activation_fn = activation_fn
    
    def forward(self, input):
        assert(len(input) == self.dim_in)
        return self.activation_fn(sum([xi*wi for xi, wi in zip(input, self.weights)]) + (self.bias if self.bias is not None else 0))
    
    def parameters(self):
        if self._parameters is None:
            self._parameters = self.weights + ([] if self.bias is None else [self.bias])
        return self._parameters


class Linear(Module):
    def __init__(self, dim_in, dim_out, bias=True, activation_fn=F.tanh):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.neurons = [_Neuron(dim_in=dim_in, bias=bias, activation_fn=activation_fn) for _ in range(dim_out)]
    
    def forward(self, input):
        assert(len(input) == self.dim_in)
        return [n(input) for n in self.neurons]
    
    def parameters(self):
        if self._parameters is None:
            params = []
            for n in self.neurons:
                params.extend(n.parameters())
            self._parameters = params
        
        return self._parameters