import os
import shutil
import tempfile
import unittest
import warnings

import numpy
import torch
from torch import nn

import chainer
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer import training


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.l = nn.Linear(1, 3)
        self.double()

    def forward(self, x):
        return self.l(x).mean()


class Dataset(chainer.dataset.DatasetMixin):

    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def get_example(self, i):
        return torch.DoubleTensor([self.values[i]])


class TestFailOnNonNumber(unittest.TestCase):

    def setUp(self):
        self.n_data = 4
        self.n_epochs = 3

        self.model = Model()
        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.dataset = Dataset([i for i in range(self.n_data)])
        self.iterator = chainer.iterators.SerialIterator(
            self.dataset, 1, shuffle=False)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def prepare(self, dirname='test', device=None):
        outdir = os.path.join(self.temp_dir, dirname)
        self.updater = training.updaters.StandardUpdater(
            self.iterator, self.optimizer, self.model, device=device)
        self.trainer = training.Trainer(
            self.updater, (self.n_epochs, 'epoch'), out=outdir)
        self.trainer.extend(training.extensions.FailOnNonNumber())

    def test_trainer(self):
        self.prepare(dirname='test_trainer')
        self.trainer.run()

    def test_nan(self):
        self.prepare(dirname='test_nan')
        self.model.l.weight[1, 0] = numpy.nan
        with self.assertRaises(RuntimeError):
            self.trainer.run(show_loop_exception_msg=False)

    def test_inf(self):
        self.prepare(dirname='test_inf')
        self.model.l.weight[2, 0] = numpy.inf
        # Ignore RuntimeWarning when using Adam on CPU
        with warnings.catch_warnings(), self.assertRaises(RuntimeError):
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            self.trainer.run(show_loop_exception_msg=False)


testing.run_module(__name__, __file__)
