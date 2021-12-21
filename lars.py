"""
This is from https://github.com/JosephChenHub/pytorch-lars.
"""

import torch
from torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")

    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large batch training of convolutional networks with layer-wise adaptive rate scaling. ICLR'18:
        https://openreview.net/pdf?id=rJ4uaX2aW

    The LARS algorithm can be written as
    .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + (1.0 - \mu) * (g_{t} + \beta * w_{t}), \\
                w_{t+1} & = w_{t} - lr * ||w_{t}|| / ||v_{t+1}|| * v_{t+1},
            \end{aligned}
    where :math:`w`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, params, lr, momentum=.9,
                 weight_decay=.0005, dampening = 0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}"
                             .format(weight_decay))
        #if eta < 0.0:
        #    raise ValueError("Invalid eta value:{}".format(eta))

        defaults = dict(lr=lr, momentum = momentum,
                        weight_decay = weight_decay,
                        dampening = dampening)

        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()


        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            dampening = group['dampening']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                # gradient
                d_p = p.grad.data
                weight_norm = torch.norm(p.data)

                # update the velocity
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                # l2 regularization
                if weight_decay != 0:
                    d_p.add_(p, alpha=weight_decay)

                buf.mul_(momentum).add_(d_p, alpha = 1.0 - dampening)
                v_norm = torch.norm(buf)

                local_lr = lr * weight_norm / (1e-6 + v_norm)

                # Update the weight
                p.add_(buf, alpha = -local_lr)


        return loss



