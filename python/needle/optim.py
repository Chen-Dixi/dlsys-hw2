"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p not in self.u:
                ut = ndl.init.zeros(*p.grad.shape, device=p.device, dtype=p.dtype, requires_grad=False)
                t = 0
            else:
                ut, t = self.u[p]
            param_data = p.data
            ut_1 = self.momentum * ut + (1-self.momentum) * (p.grad.data + self.weight_decay * param_data)
            
            # 迭代下一次
            self.u[p] = (ut_1, t+1)
            p.data = param_data - self.lr * ut_1
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        # 把t用字典代替
        self.t = {}
        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p not in self.m:
                mt = ndl.init.zeros(*p.grad.shape, device=p.device, dtype=p.dtype, requires_grad=False)
            else:
                mt = self.m[p]
            
            if p not in self.v:
                vt = ndl.init.zeros(*p.grad.shape, device=p.device, dtype=p.dtype, requires_grad=False)
            else:
                vt = self.v[p]

            if p not in self.t:
                t = 0
            else:
                t = self.t[p]
            
            grad = p.grad.data + self.weight_decay * p.data
            
            mt_1 = self.beta1 * mt + (1-self.beta1) * grad
            vt_1 = self.beta2 * vt + (1-self.beta2) * (grad ** 2)
            t_1 = t + 1
            self.m[p] = mt_1
            self.v[p] = vt_1
            self.t[p] = t_1
            # bias correction

            m_hat = mt_1 / (1 - self.beta1 ** t_1)
            v_hat = vt_1 / (1 - self.beta2 ** t_1)
            p.data = p.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)
        ### END YOUR SOLUTION
