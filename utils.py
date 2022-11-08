from collections.abc import Iterable


def argmax(input):
    m, id = None, None
    for i, item in enumerate(input):
        if id is None or item.value > m:
            m = item
            id = i
    return id

def one_hot(input, num_classes = None):
    
    if isinstance(input, Iterable):
        if num_classes is None:
            num_classes = max(input) + 1
        ret = []
        for i in input:
            ret.append([1 if j==i else 0 for j in range(num_classes)])

        return ret
    
    return [1 if j==input else 0 for j in range(input+1 if num_classes is None else num_classes)]