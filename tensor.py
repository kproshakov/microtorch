from math import log, e

class Tensor:
    def __init__(self, value=0, parents=None, label='-', op='init'):
        self.value = value
        self.grad = 0
        self._calc_grad = lambda : None
        self.parents = parents
        self.label = label
        self.op = op
    
    def backward(self):
        self.grad = 1
        def _get_children(t):
            if t is None:
                return []

            if t.parents is None:
                return [t]
            else:
                parents_list = []
                for tc in t.parents:
                    parents_list.extend(_get_children(tc))
                return parents_list + [t]
        
        order = _get_children(self)
        for t in reversed(order):
            t._calc_grad()

    # def exp(self):
    #     return e**self
        

    def __repr__(self):
        return f'Tensor(value={self.value})'
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(value=other, label=str(other))
        ret = Tensor(value=self.value + other.value, parents=(self, other), label = '(' + self.label + " + " + other.label + ')', op='add')
        def _calc_grad():
            self.grad += ret.grad
            other.grad += ret.grad
            # print(f'New grad for {self.label}: {self.grad}\nNew grad for {other.label}: {other.grad}\n\n')

        ret._calc_grad = _calc_grad
        return ret
    
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(value=other, label=str(other))
        ret = Tensor(value = self.value * other.value, parents=(self, other), label = '(' + self.label + " * " + other.label + ')', op='mul')
        def _calc_grad():
            self.grad += ret.grad*other.value
            other.grad += ret.grad*self.value
            # print(f'New grad for {self.label}: {self.grad}\nNew grad for {other.label}: {other.grad}\n\n')

        ret._calc_grad = _calc_grad
        return ret

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-1*other)
    
    def __rsub__(self, other):
        return (-1*self) + other

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(value=other, label=str(other))
        ret = Tensor(value=self.value**other.value, parents=(self, other), label = '(' + self.label + " ** " + other.label + ')', op='pov')
        def _calc_grad():
            self.grad += ret.grad*other.value*(self.value**(other.value-1))
            # other.grad += ret.grad*log(self.value)*ret.value
            # print(f'New grad for {self.label}: {self.grad}\nNew grad for {other.label}: {other.grad}\n\n')

        ret._calc_grad = _calc_grad
        return ret

    def __rpow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(value=other, label=str(other))
        ret = Tensor(value=other.value**self.value, parents=(self, other), label = '(' + other.label + " ** " + self.label + ')', op='rpov')
        def _calc_grad():
            # self.grad += ret.grad*log(other.value)*ret.value
            other.grad += ret.grad*self.value*(other.value**(self.value-1))
            # print(f'New grad for {self.label}: {self.grad}\nNew grad for {other.label}: {other.grad}\n\n')
            
        ret._calc_grad = _calc_grad
        return ret

    def __truediv__(self, other):
        return self * (other**-1)
    
    def __rtruediv__(self, other):
        return other * (self**-1)
    
    def __neg__(self):
        return -1*self
    
    def __eq__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(value=other, label=str(other))
        return self.value == other.value
    
    def __gt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(value=other, label=str(other))
        return self.value > other.value
    
    def __lt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(value=other, label=str(other))
        return self.value < other.value
    
    def __neq__(self, other):
        return not self == other
    
    def __le__(self, other):
        return not self > other
    
    def __ge__(self, other):
        return not self < other