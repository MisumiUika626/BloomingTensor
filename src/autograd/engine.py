class Value:
    def __init__(self,data,_children=(),_op=""):
        self.data=data
        self.grad=0.0 #
        self._prev=set(_children)
        self._op=_op
        self._backward=lambda:None #
    def __add__(self,other):
        out=Value(self.data+other.data,
                  (self,other),
                  "+")
        def _backward():
            self.grad+=out.grad
            other.grad+=out.grad
        out._backward=_backward
        return out
    def __mul__(self,other):
            out=Value(self.data*other.data,
                      (self,other),
                      "*")
            def _backward():
                self.grad+=out.grad*other.data
                other.grad+=out.grad*self.data
            out._backward=_backward
            return out
    def backward(self):
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
a = Value(2.0)
b = Value(3.0)
c = a + b

c.backward()

print(c.data)
print(c.grad)
print(a.grad)
print(b.grad)
              
              
              
              
         

