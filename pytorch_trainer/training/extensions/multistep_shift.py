from __future__ import division

from pytorch_trainer.training import extension


class MultistepShift(extension.Extension):

    """Trainer extension to shift an optimizer attribute in several steps.

    This extension changes an optimizer attribute in several steps, every step
    the attribute will multiply a factor ``gamma``.

    For example, suppose that this extension is called at every iteration,
    and ``init = x``, ``gamma = y``, ``step_value = [s1, s2, s3]``.
    Then during the iterations from 0 to (s1 - 1), the attr will be ``x``.
    During the iterations from s1 to (s2 - 1), the attr will be ``x * y``.
    During the iterations from s2 to (s3 - 1), the attr will be ``x * y * y``.
    During the iterations after s3, the attr will be ``x * y * y * y``.

    This extension is also called before the training loop starts by default.

    Args:
        attr (str): Name of the attribute to shift.
        init (float): Initial value of the attribute. If it is ``None``, the
            extension extracts the attribute at the first call and uses it as
            the initial value.
        gamma (float): The factor which the attr will mutiply at the beginning
            of each step.
        step_value (tuple): The first iterations of each step.
        optimizer (~chainer.Optimizer): Target optimizer to adjust the
            attribute. If it is ``None``, the main optimizer of the updater is
            used.

    """

    def __init__(self, attr, gamma, step_value, init, optimizer=None):
        self._attr = attr
        self._gamma = gamma
        self._step_value = step_value
        self._init = init
        self._optimizer = optimizer
        self._stepvalue_size = len(step_value)
        self._current_step = 0
        self._t = 0

    def initialize(self, trainer):
        optimizer = self._optimizer or trainer.updater.get_optimizer('main')
        if self._init is None:
            self._init = optimizer.param_groups[0][self._attr]
        else:
            for param_group in optimizer.param_groups:
                param_group[self._attr] = self._init

    def __call__(self, trainer):
        self._t += 1
        optimizer = self._optimizer or trainer.updater.get_optimizer('main')
        if (self._current_step < self._stepvalue_size and
                self._t >= self._step_value[self._current_step]):
            self._current_step += 1
        value = self._init * pow(self._gamma, self._current_step)
        for param_group in optimizer.param_groups:
            param_group[self._attr] = value

    def state_dict(self):
        return {"t": self._t, "current_step": self._current_step}

    def load_state_dict(self, state_dict):
        self._t = state_dict["t"]
        self._current_step = state_dict["current_step"]
