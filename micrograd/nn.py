# M4：从零实现你的神经网络库（Neuron / Layer / MLP）
import random
from micrograd.engine import Value

class Module():
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, inputs, nonlin = True):
        self.inputs = inputs
        self.nonlin = nonlin
        self.l_w = [Value(random.uniform(-1, 1) * 0.01) for _ in range(inputs)]
        self.b = Value(0)

    def __call__(self, l_in):
        res = sum((wi*xi for wi, xi in zip(self.l_w, l_in)), self.b)
        return res.relu() if self.nonlin else res

    def parameters(self):
        return self.l_w + [self.b]

class Layer(Module):
    def __init__(self, inputs, n_neu, nonlin = True):
        self.inputs = inputs
        self.n_neu = n_neu
        self.nonlin = nonlin
        self.l_neu = [Neuron(inputs, nonlin) for _ in range(n_neu)]

    def __call__(self, in_l):
        if(self.n_neu == 1):
            return self.l_neu[0](in_l)
        
        res = []
        for i in range(self.n_neu):
            res.append(self.l_neu[i](in_l))

        return res

    def parameters(self):
        res = []
        for i in range(self.n_neu):
            res += self.l_neu[i].parameters()

        return res

class MLP(Module):
    def __init__(self, inputs, l_pra):
        self.inputs = inputs
        self.l_pra = l_pra.copy()
        self.l_lay = []
        wid = inputs
        for i,nout in enumerate(l_pra):
            self.l_lay.append(Layer(wid, nout, i!=len(l_pra) - 1))
            wid = nout

    def __call__(self, in_l):
        inputs = in_l.copy()
        for i in range(len(self.l_lay)):
            inputs = self.l_lay[i](inputs)

        return inputs

    def parameters(self):
        res = []
        for i in range(len(self.l_lay)):
            res += self.l_lay[i].parameters()

        return res