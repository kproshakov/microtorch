from tensor import Tensor
from math import e


def relu(input):
    ret = Tensor(value=max(0, input.value), label='relu(' + input.label + ')', parents = tuple([input]), op = 'relu')
    def _calc_grad():
        input.grad += ret.grad if input > 0 else 0
    ret._calc_grad = _calc_grad
    return ret


def tanh(input):
    ret = Tensor(value=((e**input.value - e**(-input))/(e**(input) + e**(-input))), op='tanh', parents=tuple([input]))
    def _calc_grad():
        input.grad += ret.grad*(1-ret.value**2)
    ret._calc_grad = _calc_grad
    ret.label = 'tanh(' + input.label + ')'
    return ret

def exp(input):
    return e ** input

def sigmoid(input):
    return (1+exp(-input))**-1