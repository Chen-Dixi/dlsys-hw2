"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(
            in_features,
            out_features,
            device=None, dtype=dtype))
        self.need_bias = bias
        if bias:
            self.bias = Parameter(init.kaiming_uniform(
                out_features, 
                1,
                device=device, dtype=dtype).reshape((1, out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        logit = ops.matmul(X, self.weight)

        if self.need_bias:
            logit = ops.add(logit, self.bias.broadcast_to(logit.shape))

        return logit
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        in_shape = logits.shape
        n_dim = in_shape[-1]
        zy_sum = (logits * init.one_hot(n_dim, y, device=y.device, dtype=logits.dtype)).sum()
        logexp_sum = ops.logsumexp(logits, axes=-1).sum()
        return (logexp_sum - zy_sum) / in_shape[0]
        ### END YOUR SOLUTION
    
    # def forward(self, logits: Tensor, y: Tensor):
    #     ### BEGIN YOUR SOLUTION
    #     exp_sum = ops.logsumexp(logits, axes=(1, )).sum()
    #     z_y_sum = (logits * init.one_hot(logits.shape[1], y)).sum()
    #     return (exp_sum - z_y_sum) / logits.shape[0]
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        
        batch_size = x.shape[0]
        if self.training:
            batch_mean = x.sum(axes=0) / batch_size # (dim,) batch的均值
            x_mean = x - batch_mean.broadcast_to(x.shape) # (b, dim)
            std = ((x_mean ** 2).sum(axes=0) / batch_size) # (dim, ) batch的方差
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * std.detach()

        else:
            x_mean = x - self.running_mean.broadcast_to(x.shape)
            std = self.running_var
        
        normed = x_mean / ((std + self.eps) ** 0.5).broadcast_to(x.shape) # 分布标准化之后 (b, dim)
        return self.weight.broadcast_to(x.shape) * normed + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = ops.reshape(x.sum(axes=-1) / self.dim, (-1, 1)) # (batch, 1)
        x_mean = x - mean.broadcast_to(x.shape) # (batch, n)
        std = (((x_mean ** 2).sum(axes=-1).reshape((-1,1)) / self.dim) + self.eps) ** 0.5 # # (batch, 1)
        normed = x_mean / std.broadcast_to(x.shape)
        # 1 维张量 (4,)可以被broadcast_to为(b, 4),但不能broadcast_to(4, b)。扩展的维度只能是前一个
        return self.weight.broadcast_to(x.shape) * normed + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mask = init.randb(*x.shape, p=1 - self.p)
        if self.training:
            x_mask = x * mask
            return x_mask / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION