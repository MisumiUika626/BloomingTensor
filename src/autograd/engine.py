class Value:
    def __init__(self,data,_children=(),_op=""):
        self.data=data
        self.grad=0.0
        self._prev=set(_children)
        self._op=_op
        self._backward=lambda:None
    
