import copy
import io
import unittest

import torch

import chainer
from chainer import testing


class DummyTrainer(object):

    def __init__(self):
        self.elapsed_time = 0


class TestTimeTrigger(unittest.TestCase):

    def setUp(self):
        self.trigger = chainer.training.triggers.TimeTrigger(1)
        self.trainer = DummyTrainer()

    def test_call(self):
        assert not self.trigger(self.trainer)
        self.trainer.elapsed_time = 0.9
        assert not self.trigger(self.trainer)

        # first event is triggerred on time==1.0
        self.trainer.elapsed_time = 1.2
        assert self.trigger(self.trainer)

        self.trainer.elapsed_time = 1.3
        assert not self.trigger(self.trainer)

        # second event is triggerred on time==2.0, and is not on time==2.2
        self.trainer.elapsed_time = 2.1
        assert self.trigger(self.trainer)

    def test_resume(self):
        self.trainer.elapsed_time = 1.2
        self.trigger(self.trainer)
        assert self.trigger._next_time == 2.0

        f = io.BytesIO()
        torch.save(self.trigger.state_dict(), f)

        trigger = chainer.training.triggers.TimeTrigger(1)
        trigger.load_state_dict(torch.load(io.BytesIO(f.getvalue())))
        assert trigger._next_time == 2.0


testing.run_module(__name__, __file__)
