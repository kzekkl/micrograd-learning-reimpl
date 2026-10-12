# M2：从零实现你的 Value 类（你的引擎从这里开始）
class Value:#节点类

    def __init__(self, data, grad = 0, prev = (), op = ""):
        self.data = data    #数值，用于前向传播
        self.grad = grad    #梯度，反向传播给出
        self.prev = prev
        self.op = op
        self._backward = lambda: None

    #1.1 重载+
    def __add__(self, other):
        if  not isinstance(other,Value):
            other = Value(other)

        data_res = self.data + other.data
        res = Value(data_res, 0, (self,other), "+ ")
        def _backward():
            self.grad += res.grad
            other.grad += res.grad

        res._backward = _backward
        return res

    #1.2 左加
    def __radd__(self, other):
        return self.__add__(other)

    #2.1 重载*
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        data_res = self.data * other.data
        res = Value(data_res, 0, (self,other), "* ")
        def _backward():
            self.grad += other.data * res.grad
            other.grad += self.data * res.grad

        res._backward = _backward
        return res

    #2.2 处理数字左乘
    def __rmul__(self, other):
        return self.__mul__(other)

    #3. 重载**
    def __pow__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return Value(self.data**other)

    #4.1 重载/
    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Value(self.data/other)

    #4.2 数字左c除
    def __rtruediv__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Value(other/self.data)

    #5. 定义relu
    def relu(self):
        data_res = self.data if self.data>0 else 0
        res = Value(data_res, 0, (self,), "re ")
        def _backward():
            self.grad += res.grad if self.data>0 else 0

        res._backward = _backward
        return res

    #6. 反向传播
    def backward(self):
        self.grad = 1
        seen = set()
        order = []

        def collect(v):
            if v in seen:
                return
            
            seen.add(v)
            for p in v.prev:
                collect(p)

            order.append(v)

        collect(self)
        for node in reversed(order):
            node._backward()

    #7. 取负
    def __neg__(self):
        return Value(-self.data)