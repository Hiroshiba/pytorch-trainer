import copy
import os
import subprocess
import sys
import unittest

import numpy

import chainer
from chainer.backends import cuda
import chainer.functions.math.minmax
from chainer import initializers
import chainer.reporter
from chainer import testing
from chainer.testing import attr
import chainer.training.updaters.multiprocess_parallel_updater as mpu


class SimpleNet(chainer.Chain):
    insize = 5

    def __init__(self, dtype=numpy.float32):
        super(SimpleNet, self).__init__()
        self.dtype = dtype
        W = initializers.HeNormal(1 / numpy.sqrt(2), self.dtype)
        bias = initializers.Zero(self.dtype)
        with self.init_scope():
            self.conv = chainer.links.Convolution2D(2, 2, 3, initialW=W,
                                                    initial_bias=bias)
            self.fc = chainer.links.Linear(18, 2, initialW=W,
                                           initial_bias=bias)
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        h = chainer.functions.relu(self.conv(x))
        y = self.fc(h)

        self.loss = chainer.functions.softmax_cross_entropy(y, t)
        self.accuracy = chainer.functions.accuracy(y, t)

        return self.loss


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float16],
}))
class TestGatherScatter(unittest.TestCase):

    def test_gather_grads_raise_on_cpu(self):
        model = SimpleNet(dtype=self.dtype)
        with self.assertRaises(RuntimeError):
            mpu.gather_grads(model)

    def test_gather_params_raise_on_cpu(self):
        model = SimpleNet(dtype=self.dtype)
        with self.assertRaises(RuntimeError):
            mpu.gather_params(model)


def _run_test_snippet(name, *args):
    script_path = os.path.join(
        os.path.dirname(__file__), 'snippets/{}'.format(name))
    proc = subprocess.Popen(
        (sys.executable, script_path) + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdoutdata, stderrdata = proc.communicate()
    ret = proc.returncode
    return (ret, stdoutdata, stderrdata)


class TestRawArray(unittest.TestCase):

    @attr.gpu
    @unittest.skipUnless(mpu.MultiprocessParallelUpdater.available(),
                         'MultiprocessParallelUpdater is not available.')
    def test_update_uses_raw_array(self):
        ret, stdoutdata, stderrdata = _run_test_snippet(
            'raw_array.py', '@cupy:0')
        assert ret == 0, (
            '[stdout]:{!r}\n'
            '[stderr]:{!r}'.format(stdoutdata, stderrdata))


class TestChildReporter(unittest.TestCase):

    def check_with_devices(self, n_devices):
        devices_str = ','.join([
            '@cupy:{}'.format(device_id) for device_id in range(n_devices)])
        ret, stdoutdata, stderrdata = _run_test_snippet(
            'child_reporter.py', devices_str)
        assert ret == 0, (
            '[stdout]:{!r}\n'
            '[stderr]:{!r}'.format(stdoutdata, stderrdata))

    @attr.gpu
    @unittest.skipUnless(mpu.MultiprocessParallelUpdater.available(),
                         'MultiprocessParallelUpdater is not available.')
    def test_single_device(self):
        self.check_with_devices(1)

    @attr.multi_gpu(2)
    @unittest.skipUnless(mpu.MultiprocessParallelUpdater.available(),
                         'MultiprocessParallelUpdater is not available.')
    def test_multi_device(self):
        self.check_with_devices(2)


class TestCUDAContext(unittest.TestCase):

    @attr.gpu
    @unittest.skipUnless(mpu.MultiprocessParallelUpdater.available(),
                         'MultiprocessParallelUpdater is not available.')
    def test_cuda_init(self):
        ret, stdoutdata, stderrdata = _run_test_snippet('cuda_init.py')
        assert ret == 0, (
            '[stdout]:{!r}\n'
            '[stderr]:{!r}'.format(stdoutdata, stderrdata))


class TestDevicesByDeviceIds(unittest.TestCase):

    @attr.gpu
    @unittest.skipUnless(mpu.MultiprocessParallelUpdater.available(),
                         'MultiprocessParallelUpdater is not available.')
    def test_devices_by_device_ids_array(self):
        # Test passing devices to MultiprocessParallelUpdater by their ids.
        ret, stdoutdata, stderrdata = _run_test_snippet(
            'raw_array.py', '0')
        assert ret == 0, (
            '[stdout]:{!r}\n'
            '[stderr]:{!r}'.format(stdoutdata, stderrdata))


testing.run_module(__name__, __file__)
