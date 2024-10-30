import torch
from torch.optim.optimizer import Optimizer


class UniversalSGD(Optimizer):
    """Implements Universal Stochastic Gradient Method
    Universal Stochastic Gradient Method (Algorithm 4.1) was proposed by Anton Rodomanov et.al https://arxiv.org/pdf/2402.03210
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        D (float): size of the ball that includes starting point and solution.
    """
    MONOTONE = False
    SKIP_TEST_LOGREG = True

    def __init__(self, params, D: float = 1., passive_start: bool = True, verbose: bool = True, testing: bool = False):
        if D <= 0:
            raise ValueError(f"Invalid set size: D = {D}")

        super().__init__(params, dict(D=D))
        self.D = D
        self.Hk = 0.
        self.passive_start = passive_start
        self.verbose = verbose
        self.testing = testing
        self.iter_count = 0
        if len(self.param_groups) != 1:
            raise ValueError("Method doesn't support per-parameter options "
                             "(parameter groups)")
        group = self.param_groups[0]
        params = group['params']

        # Initialization of intermediate points
        for p in params:
            state = self.state[p]
            state['x0'] = p.detach().clone()
            state['x'] = state['x0'].clone()
            state['grad'] = state['x0'].zero_().clone()
            state['x_av'] = state['x0'].clone()

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): a closure that reevaluates the model and returns the loss
        """
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        params = group['params']

        if self.iter_count == 0:
            grad = torch.autograd.grad(closure(), list(params))
            with torch.no_grad():
                norm_sq = 0.
                for g in grad:
                    norm_sq += g.square().sum()
                norm_grad = norm_sq ** 0.5
                h = self.D / norm_grad
                rk1 = 0.
                for p, g in zip(params, grad):
                    dif = g * h
                    p.sub_(dif)
                    rk1 += dif.square().sum()
                rk1 = rk1 ** 0.5

            grad2 = torch.autograd.grad(closure(), list(params))

            with torch.no_grad():
                beta_k1 = 0.
                for p, g1, g2 in zip(params, grad, grad2):
                    state = self.state[p]
                    beta_k1 += (p - state['x0']).mul(g2 - g1).sum()
                self.Hk += beta_k1 / (self.D ** 2 + rk1 ** 2 / 2)
                if self.passive_start:
                    for p, g in zip(params, grad):
                        state = self.state[p]
                        state['grad'] = g.detach().clone()
                        p.zero_().add_(state['x0'])
                        self.iter_count += 1
                else:
                    for p, g in zip(params, grad2):
                        state = self.state[p]
                        state['grad'] = g.detach().clone()
                        state['x'].zero_().add_(p)
                        self.iter_count += 2
                        state['x_av'] = (state['x_av'] + state['x']) / 2.
                        p.zero_().add_(state['x_av'])
        else:
            with torch.no_grad():
                h = 1. / self.Hk
                rk1 = 0.
                rk0 = 0.
                for p in params:
                    state = self.state[p]
                    dif = h * state['grad']
                    p.zero_().add_(state['x'] - dif)
                    rk1 += dif.square().sum()
                    rk0 += (p - state['x0']).square().sum()
                rk1 = rk1 ** 0.5
                rk0 = rk0 ** 0.5
                if rk0 > self.D:
                    for p in params:
                        state = self.state[p]
                        dif2 = p - state['x0']
                        p.zero_().add_(state['x0']).add_(self.D * dif2 / rk0)

            grad = torch.autograd.grad(closure(), list(params))

            with torch.no_grad():
                beta_k1 = 0.
                for p, g2 in zip(params, grad):
                    state = self.state[p]
                    beta_k1 += (p - state['x']).mul(g2 - state['grad']).sum()

                self.Hk += max((beta_k1 - rk1 ** 2 * self.Hk / 2) / (self.D ** 2 + rk1 ** 2 / 2), 0.)

                for p, g2 in zip(params, grad):
                    state = self.state[p]
                    state['grad'] = g2.detach().clone()
                    state['x'].zero_().add_(p)
                    state['x_av'] = (state['x_av'] * self.iter_count + state['x']) / (self.iter_count + 1.)
                    self.iter_count += 1
                    p.zero_().add_(state['x_av'])
        return None