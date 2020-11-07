import unittest

import numpy
import torch
from torch import nn

import pytorch_trainer
from pytorch_trainer import dataset
from pytorch_trainer import iterators
from pytorch_trainer import testing
from pytorch_trainer.training import extensions


class DummyModel(nn.Module):

    def __init__(self, test):
        super(DummyModel, self).__init__()
        self.args = []
        self.test = test
        self.double()

    def forward(self, x):
        self.args.append(x)
        pytorch_trainer.report({'loss': x.sum()}, self)


class DummyModelTwoArgs(nn.Module):

    def __init__(self, test):
        super(DummyModelTwoArgs, self).__init__()
        self.args = []
        self.test = test
        self.double()

    def forward(self, x, y):
        self.args.append((x, y))
        pytorch_trainer.report({'loss': x.sum() + y.sum()}, self)


class DummyIterator(dataset.Iterator):

    def __init__(self, return_values):
        self.iterator = iter(return_values)
        self.finalized = False
        self.return_values = return_values

    def reset(self):
        self.iterator = iter(self.return_values)

    def __next__(self):
        return next(self.iterator)

    def finalize(self):
        self.finalized = True


class DummyConverter(object):

    def __init__(self, return_values):
        self.args = []
        self.iterator = iter(return_values)

    def __call__(self, batch, device):
        self.args.append({'batch': batch, 'device': device})
        return next(self.iterator)


class TestEvaluator(unittest.TestCase):

    def setUp(self):
        self.data = [
            torch.empty(3, 4).uniform_(-1, 1) for _ in range(2)]
        self.batches = [
            torch.empty(2, 3, 4).uniform_(-1, 1)
            for _ in range(2)]

        self.iterator = DummyIterator(self.data)
        self.converter = DummyConverter(self.batches)
        self.target = DummyModel(self)
        self.evaluator = extensions.Evaluator(
            self.iterator, self.target, converter=self.converter)
        self.expect_mean = torch.stack([torch.sum(x) for x in self.batches]).mean()

    def test_evaluate(self):
        reporter = pytorch_trainer.Reporter()
        reporter.add_observer('target', self.target)
        with reporter:
            mean = self.evaluator.evaluate()

        # No observation is reported to the current reporter. Instead the
        # evaluator collect results in order to calculate their mean.
        self.assertEqual(len(reporter.observation), 0)

        # The converter gets results of the iterator.
        self.assertEqual(len(self.converter.args), len(self.data))
        for i in range(len(self.data)):
            numpy.testing.assert_array_equal(
                self.converter.args[i]['batch'], self.data[i])
            self.assertIsNone(self.converter.args[i]['device'])

        # The model gets results of converter.
        self.assertEqual(len(self.target.args), len(self.batches))
        for i in range(len(self.batches)):
            numpy.testing.assert_array_equal(
                self.target.args[i], self.batches[i])

        self.assertAlmostEqual(mean['target/loss'], self.expect_mean, places=4)

        self.evaluator.finalize()
        self.assertTrue(self.iterator.finalized)

    def test_call(self):
        mean = self.evaluator()
        # 'main' is used by default
        self.assertAlmostEqual(mean['main/loss'], self.expect_mean, places=4)

    def test_evaluator_name(self):
        self.evaluator.name = 'eval'
        mean = self.evaluator()
        # name is used as a prefix
        self.assertAlmostEqual(
            mean['eval/main/loss'], self.expect_mean, places=4)

    def test_current_report(self):
        reporter = pytorch_trainer.Reporter()
        with reporter:
            mean = self.evaluator()
        # The result is reported to the current reporter.
        self.assertEqual(reporter.observation, mean)

@testing.parameterize(
    {'device': 'cpu'})
class TestEvaluatorTupleData(unittest.TestCase):

    def setUp(self):
        self.data = [
            torch.empty(3, 4).uniform_(-1, 1) for _ in range(2)]
        self.batches = [
            (torch.empty(2, 3, 4).uniform_(-1, 1),
             torch.empty(2, 3, 4).uniform_(-1, 1))
            for _ in range(2)]

    def prepare(self, data, batches, device):
        iterator = DummyIterator(data)
        converter = DummyConverter(batches)
        target = DummyModelTwoArgs(self)
        evaluator = extensions.Evaluator(
            iterator, target, converter=converter, device=device)
        return iterator, converter, target, evaluator

    def test_evaluate(self):
        data = self.data
        batches = self.batches
        device = self.device

        iterator, converter, target, evaluator = (
            self.prepare(data, batches, device))

        reporter = pytorch_trainer.Reporter()
        reporter.add_observer('target', target)
        with reporter:
            mean = evaluator.evaluate()

        # The converter gets results of the iterator and the device number.
        self.assertEqual(len(converter.args), len(data))
        expected_device_arg = self.device

        for i in range(len(data)):
            numpy.testing.assert_array_equal(
                converter.args[i]['batch'].cpu().numpy(), self.data[i])
            self.assertEqual(converter.args[i]['device'].type, expected_device_arg)

        # The model gets results of converter.
        self.assertEqual(len(target.args), len(batches))
        for i in range(len(batches)):
            self.assertEqual(target.args[i], self.batches[i])

        expect_mean = torch.stack([torch.stack(x).sum() for x in self.batches]).mean()
        self.assertAlmostEqual(
            mean['target/loss'].cpu().numpy(), expect_mean.cpu().numpy(), places=4)


class TestEvaluatorDictData(unittest.TestCase):

    def setUp(self):
        self.data = range(2)
        self.batches = [
            {'x': torch.empty(2, 3, 4).uniform_(-1, 1),
             'y': torch.empty(2, 3, 4).uniform_(-1, 1)}
            for _ in range(2)]

        self.iterator = DummyIterator(self.data)
        self.converter = DummyConverter(self.batches)
        self.target = DummyModelTwoArgs(self)
        self.evaluator = extensions.Evaluator(
            self.iterator, self.target, converter=self.converter)

    def test_evaluate(self):
        reporter = pytorch_trainer.Reporter()
        reporter.add_observer('target', self.target)
        with reporter:
            mean = self.evaluator.evaluate()

        # The model gets results of converter.
        self.assertEqual(len(self.target.args), len(self.batches))
        for i in range(len(self.batches)):
            numpy.testing.assert_array_equal(
                self.target.args[i][0], self.batches[i]['x'])
            numpy.testing.assert_array_equal(
                self.target.args[i][1], self.batches[i]['y'])

        expect_mean = torch.stack(
            [x['x'].sum() + x['y'].sum() for x in self.batches]).mean()
        self.assertAlmostEqual(mean['target/loss'].cpu().numpy(), expect_mean.cpu().numpy(), places=4)


class TestEvaluatorWithEvalFunc(unittest.TestCase):

    def setUp(self):
        self.data = [
            torch.empty(3, 4).uniform_(-1, 1) for _ in range(2)]
        self.batches = [
            torch.empty(2, 3, 4).uniform_(-1, 1)
            for _ in range(2)]

        self.iterator = DummyIterator(self.data)
        self.converter = DummyConverter(self.batches)
        self.target = DummyModel(self)
        self.evaluator = extensions.Evaluator(
            self.iterator, {}, converter=self.converter,
            eval_func=self.target)

    def test_evaluate(self):
        reporter = pytorch_trainer.Reporter()
        reporter.add_observer('target', self.target)
        with reporter:
            self.evaluator.evaluate()

        # The model gets results of converter.
        self.assertEqual(len(self.target.args), len(self.batches))
        for i in range(len(self.batches)):
            numpy.testing.assert_array_equal(
                self.target.args[i], self.batches[i])


@testing.parameterize(*testing.product({
    'repeat': [True, False],
    'iterator_class': [iterators.SerialIterator,
                       iterators.MultiprocessIterator,
                       iterators.MultithreadIterator]
}))
class TestEvaluatorRepeat(unittest.TestCase):

    def test_user_warning(self):
        dataset = torch.ones((4, 6))
        iterator = self.iterator_class(dataset, 2, repeat=self.repeat)
        if self.repeat:
            with testing.assert_warns(UserWarning):
                extensions.Evaluator(iterator, {})


class TestEvaluatorProgressBar(unittest.TestCase):

    def setUp(self):
        self.data = [
            torch.empty(3, 4).uniform_(-1, 1) for _ in range(2)]

        self.iterator = iterators.SerialIterator(
            self.data, 1, repeat=False, shuffle=False)
        self.target = DummyModel(self)
        self.evaluator = extensions.Evaluator(
            self.iterator, {}, eval_func=self.target, progress_bar=True)

    def test_evaluator(self):
        reporter = pytorch_trainer.Reporter()
        reporter.add_observer('target', self.target)
        with reporter:
            self.evaluator.evaluate()


testing.run_module(__name__, __file__)
