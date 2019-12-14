import copy
import unittest

import mock
import torch
from torch import DoubleTensor
from torch import nn
from torch.optim.optimizer import Optimizer

import chainer
from chainer import dataset
from chainer import testing
from chainer import training


class DummyIterator(dataset.Iterator):

    epoch = 1
    is_new_epoch = True

    def __init__(self, next_data):
        self.finalize_called = 0
        self.next_called = 0
        self.next_data = next_data
        self.state_dict = mock.Mock()
        self.load_state_dict = mock.Mock()

    def finalize(self):
        self.finalize_called += 1

    def __next__(self):
        self.next_called += 1
        return self.next_data


class DummyOptimizer(Optimizer):

    def __init__(self, params):
        super().__init__(params, defaults={})
        self.step = mock.MagicMock()
        self.state_dict = mock.Mock()
        self.load_state_dict = mock.Mock()


class DummySerializer(chainer.Serializer):

    def __init__(self, path=None):
        if path is None:
            path = []
        self.path = path
        self.called = []

    def __getitem__(self, key):
        return DummySerializer(self.path + [key])

    def __call__(self, key, value):
        self.called.append((key, value))


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.double()

    def forward(self, x, y=None):
        return self.linear(x).mean()


class TestStandardUpdater(unittest.TestCase):

    def setUp(self):
        self.target = DummyModel()
        self.iterator = DummyIterator([(DoubleTensor([1]), DoubleTensor([2]))])
        self.optimizer = DummyOptimizer(self.target.parameters())
        self.updater = training.updaters.StandardUpdater(
            self.iterator, self.optimizer, self.target)

    def test_init_values(self):
        assert self.updater.device is None
        assert self.updater.loss_func is None
        assert self.updater.iteration == 0

    def test_epoch(self):
        assert self.updater.epoch == 1

    def test_new_epoch(self):
        assert self.updater.is_new_epoch is True

    def test_get_iterator(self):
        assert self.updater.get_iterator('main') is self.iterator

    def test_get_optimizer(self):
        assert self.updater.get_optimizer('main') is self.optimizer

    def test_get_all_optimizers(self):
        assert self.updater.get_all_optimizers() == {'main': self.optimizer}

    def test_update(self):
        self.updater.update()
        assert self.updater.iteration == 1
        assert self.updater.epoch == 1
        assert self.iterator.next_called == 1

    def test_finalizer(self):
        self.updater.finalize()
        assert self.iterator.finalize_called == 1

    def test_state_dict(self):
        state_dict = self.updater.state_dict()
        assert self.iterator.state_dict.call_count == 1
        assert self.optimizer.state_dict.call_count == 1

        self.updater.load_state_dict(copy.deepcopy(state_dict))
        assert self.iterator.load_state_dict.call_count == 1
        assert self.optimizer.load_state_dict.call_count == 1


class TestStandardUpdaterDataTypes(unittest.TestCase):
    """Tests several data types with StandardUpdater"""

    def setUp(self):
        target = DummyModel()
        self.target = mock.Mock(target)
        self.optimizer = DummyOptimizer(target.parameters())

    def test_update_tuple(self):
        iterator = DummyIterator([(DoubleTensor([1]), DoubleTensor([2]))])
        updater = training.updaters.StandardUpdater(iterator, self.optimizer, self.target)

        updater.update_core()

        assert self.target.call_count == 1
        args, kwargs = self.target.call_args
        assert len(args) == 2
        v1, v2 = args
        assert len(kwargs) == 0

        assert self.optimizer.step.call_count == 1

        assert isinstance(v1, torch.Tensor)
        assert v1 == 1
        assert isinstance(v2, torch.Tensor)
        assert v2 == 2

        assert iterator.next_called == 1

    def test_update_dict(self):
        iterator = DummyIterator([{'x': DoubleTensor([1]), 'y': DoubleTensor([2])}])
        updater = training.updaters.StandardUpdater(iterator, self.optimizer, self.target)

        updater.update_core()

        assert self.target.call_count == 1
        args, kwargs = self.target.call_args
        assert len(args) == 0
        assert set(kwargs.keys()) == {'x', 'y'}

        v1 = kwargs['x']
        v2 = kwargs['y']
        assert isinstance(v1, torch.Tensor)
        assert v1 == 1
        assert isinstance(v2, torch.Tensor)
        assert v2 == 2

        assert iterator.next_called == 1

    def test_update_var(self):
        iterator = DummyIterator([DoubleTensor([1])])
        updater = training.updaters.StandardUpdater(iterator, self.optimizer, self.target)

        updater.update_core()

        assert self.target.call_count == 1
        args, kwargs = self.target.call_args
        assert len(args) == 1
        v1, = args
        assert len(kwargs) == 0

        assert isinstance(v1, torch.Tensor)
        assert v1 == 1

        assert iterator.next_called == 1


@testing.parameterize(
    {'converter_style': 'decorator'},
    {'converter_style': 'class'})
@chainer.testing.backend.inject_backend_tests(
    ['test_converter_given_device'],
    [
        # NumPy
        {},
    ])
class TestStandardUpdaterCustomConverter(unittest.TestCase):
    """Tests custom converters of various specs"""

    def create_optimizer(self):
        target = mock.Mock()
        optimizer = mock.Mock()
        return target, optimizer

    def create_updater(self, iterator, optimizer, target, converter, device):
        return training.updaters.StandardUpdater(
            iterator, optimizer, target, converter=converter, device=device)

    def test_converter_given_device(self, backend_config):
        self.check_converter_all(backend_config.device)

    def check_converter_all(self, device):
        self.check_converter_in_arrays(device)
        self.check_converter_in_obj(device)
        self.check_converter_out_tuple(device)
        self.check_converter_out_dict(device)
        self.check_converter_out_obj(device)

    def get_converter(self, converter_func):
        if self.converter_style == 'decorator':
            @chainer.dataset.converter()
            def wrapped_converter(*args, **kwargs):
                return converter_func(*args, **kwargs)

            return wrapped_converter

        if self.converter_style == 'class':
            class MyConverter(dataset.Converter):
                def __call__(self, *args, **kwargs):
                    return converter_func(*args, **kwargs)

            return MyConverter()

        assert False

    def test_converter_type(self):
        # Ensures that new-style converters inherit from dataset.Converter.

        def converter_impl(batch, device):
            pass

        converter = self.get_converter(converter_impl)

        if self.converter_style in ('decorator', 'class'):
            assert isinstance(converter, dataset.Converter)

    def check_converter_in_arrays(self, device_arg):
        iterator = DummyIterator([(DoubleTensor([1]), DoubleTensor([2]))])
        target, optimizer = self.create_optimizer()

        called = [0]

        def converter_impl(batch, device):
            assert isinstance(batch, list)
            assert len(batch) == 1
            samples = batch[0]
            assert isinstance(samples, tuple)
            assert len(samples) == 2
            assert isinstance(samples[0], torch.Tensor)
            assert isinstance(samples[1], torch.Tensor)
            assert samples[0] == 1
            assert samples[1] == 2
            called[0] += 1
            return samples

        converter = self.get_converter(converter_impl)

        updater = self.create_updater(
            iterator, optimizer, target, converter, device_arg)
        updater.update_core()
        assert called[0] == 1

    def check_converter_in_obj(self, device_arg):
        obj1 = object()
        obj2 = object()
        iterator = DummyIterator([obj1, obj2])
        target, optimizer = self.create_optimizer()

        called = [0]

        def converter_impl(batch, device):
            assert isinstance(batch, list)
            assert len(batch) == 2
            assert batch[0] is obj1
            assert batch[1] is obj2
            called[0] += 1
            return obj1, obj2

        converter = self.get_converter(converter_impl)

        updater = self.create_updater(
            iterator, optimizer, target, converter, device_arg)
        updater.update_core()
        assert called[0] == 1

    def check_converter_out_tuple(self, device_arg):
        iterator = DummyIterator([object()])
        target, optimizer = self.create_optimizer()
        converter_out = (object(), object())

        def converter_impl(batch, device):
            return converter_out

        converter = self.get_converter(converter_impl)

        updater = self.create_updater(
            iterator, optimizer, target, converter, device_arg)
        updater.update_core()

        assert optimizer.step.call_count == 1
        args, kwargs = target.call_args
        assert len(args) == 2
        v1, v2 = args
        assert len(kwargs) == 0

        assert v1 is converter_out[0]
        assert v2 is converter_out[1]

    def check_converter_out_dict(self, device_arg):
        iterator = DummyIterator([object()])
        target, optimizer = self.create_optimizer()
        converter_out = {'x': object(), 'y': object()}

        def converter_impl(batch, device):
            return converter_out

        converter = self.get_converter(converter_impl)

        updater = self.create_updater(
            iterator, optimizer, target, converter, device_arg)
        updater.update_core()

        assert optimizer.step.call_count == 1
        args, kwargs = target.call_args
        assert len(args) == 0
        assert len(kwargs) == 2

        assert sorted(kwargs.keys()) == ['x', 'y']
        assert kwargs['x'] is converter_out['x']
        assert kwargs['y'] is converter_out['y']

    def check_converter_out_obj(self, device_arg):
        iterator = DummyIterator([object()])
        target, optimizer = self.create_optimizer()
        converter_out = object()

        def converter_impl(batch, device):
            return converter_out

        converter = self.get_converter(converter_impl)

        updater = self.create_updater(
            iterator, optimizer, target, converter, device_arg)
        updater.update_core()

        assert optimizer.step.call_count == 1
        args, kwargs = target.call_args
        assert len(args) == 1
        v1, = args
        assert len(kwargs) == 0

        assert v1 is converter_out


testing.run_module(__name__, __file__)
