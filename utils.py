from tensor import Tensor

def argmax(input):
    m, id = None, None
    for i, item in enumerate(input):
        if id is None or item.value > m:
            m = item
            id = i
    return id
