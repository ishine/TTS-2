from collections import defaultdict
import warnings
import math
import torch

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class MinStepLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, step_size, gamma=0.1, min_lr=1e-6, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        super(MinStepLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [max(group['lr'], self.min_lr) for group in self.optimizer.param_groups]
        return [max(group['lr'] * self.gamma, self.min_lr)
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** (self.last_epoch // self.step_size), self.min_lr)
                for base_lr in self.base_lrs]


class RAdam(Optimizer):
    r"""Implements RAdam optimization algorithm.
    It has been proposed in `On the Variance of the Adaptive Learning
    Rate and Beyond`__.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.RAdam(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ https://arxiv.org/abs/1908.03265
    Note:
        Reference code: https://github.com/LiyuanLucasLiu/RAdam
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )

        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if 'betas' in param and (
                    param['betas'][0] != betas[0]
                    or param['betas'][1] != betas[1]
                ):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    msg = (
                        'RAdam does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(
                        p_data_fp32, memory_format=torch.preserve_format
                    )
                    state['exp_avg_sq'] = torch.zeros_like(
                        p_data_fp32, memory_format=torch.preserve_format
                    )
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32
                    )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (
                        1 - beta2_t
                    )
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = (
                            lr
                            * math.sqrt(
                                (1 - beta2_t)
                                * (N_sma - 4)
                                / (N_sma_max - 4)
                                * (N_sma - 2)
                                / N_sma
                                * N_sma_max
                                / (N_sma_max - 2)
                            )
                            / (1 - beta1 ** state['step'])
                        )
                    else:
                        step_size = lr / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if weight_decay != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-weight_decay * lr)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(eps)
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size)

                p.data.copy_(p_data_fp32)

        return loss


class NoamAnnealing(_LRScheduler):
    def __init__(
        self, optimizer, *, d_model, warmup_steps=None, warmup_ratio=None, max_steps=None, min_lr=1e-4, last_epoch=-1
    ):
        self._normalize = d_model ** (-0.5)
        assert not (
            warmup_steps is not None and warmup_ratio is not None
        ), "Either use particular number of step or ratio"
        assert warmup_ratio is None or max_steps is not None, "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )

        step = max(1, self.last_epoch)

        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        for initial_lr in self.base_lrs:
            if initial_lr < self.min_lr:
                raise ValueError(
                    f"{self} received an initial learning rate that was lower than the minimum learning rate."
                )

        new_lrs = [self._noam_annealing(initial_lr=initial_lr, step=step) for initial_lr in self.base_lrs]
        return new_lrs

    def _noam_annealing(self, initial_lr, step):
        mult = self._normalize * min(step ** (-0.5), step * (self.warmup_steps ** (-1.5)))
        out_lr = initial_lr * mult
        if step > self.warmup_steps:
            out_lr = max(out_lr, self.min_lr)
        return out_lr


class Lookahead(Optimizer):
    r"""Implements Lookahead optimization algorithm.
    It has been proposed in `Lookahead Optimizer: k steps forward, 1
    step back`__
    Arguments:
        optimizer: base inner optimizer optimize, like Yogi, DiffGrad or Adam.
        k: number of lookahead steps (default: 5)
        alpha: linear interpolation factor. 1.0 recovers the inner optimizer.
            (default: 5)
    Example:
        >>> import torch_optimizer as optim
        >>> yogi = optim.Yogi(model.parameters(), lr=0.1)
        >>> optimizer = optim.Lookahead(yogi, k=5, alpha=0.5)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ https://arxiv.org/abs/1907.08610
    Note:
        Reference code: https://github.com/alphadl/lookahead.pytorch
    """

    def __init__(
        self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5
    ) -> None:
        if k < 0.0:
            raise ValueError('Invalid number of lookahead steps: {}'.format(k))
        if alpha < 0:
            raise ValueError(
                'Invalid linear interpolation factor: {}'.format(alpha)
            )

        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group['counter'] = 0

    def _update(self, group):
        for fast in group['params']:
            param_state = self.state[fast]
            if 'slow_param' not in param_state:
                param_state['slow_param'] = torch.clone(fast.data).detach()

            slow = param_state['slow_param']
            fast.data.mul_(self.alpha).add_(slow, alpha=1.0 - self.alpha)
            slow.data.copy_(fast)

    def step(self, closure=None):
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = self.optimizer.step(closure=closure)
        for group in self.param_groups:
            if group['counter'] == 0:
                self._update(group)
            group['counter'] = (group['counter'] + 1) % self.k
        return loss

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.
        It contains two entries:
        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containing all parameter groups
        """
        slow_state_dict = super(Lookahead, self).state_dict()
        fast_state_dict = self.optimizer.state_dict()
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'fast_state': fast_state,
            'slow_state': slow_state_dict['state'],
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict) -> None:
        r"""Loads the optimizer state.
        Arguments:
            state_dict: optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],
        }
        fast_state_dict = {
            'state': state_dict['fast_state'],
            'param_groups': state_dict['param_groups'],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def zero_grad(self) -> None:
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        self.optimizer.zero_grad()

    def __repr__(self) -> str:
        base_str = self.optimizer.__repr__()
        format_string = self.__class__.__name__ + ' ('
        format_string += '\n'
        format_string += 'k: {}\n'.format(self.k)
        format_string += 'alpha: {}\n'.format(self.alpha)
        format_string += base_str
        format_string += '\n'
        format_string += ')'
        return format_string
