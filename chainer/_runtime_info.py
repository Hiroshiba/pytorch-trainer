import platform
import sys

import numpy
import six

from torch import cuda


class _RuntimeInfo(object):

    numpy_version = None
    cuda_info = None

    def __init__(self):
        self.numpy_version = numpy.__version__
        self.platform_version = platform.platform()
        if cuda.is_available():
            self.cuda_info = cuda.device_count()
        else:
            self.cuda_info = None

    def __str__(self):
        s = six.StringIO()
        s.write('''Platform: {}\n'''.format(self.platform_version))
        s.write('''NumPy: {}\n'''.format(self.numpy_version))
        if self.cuda_info is None:
            s.write('''CUDA: Not Available\n''')
        else:
            s.write('''CUDA: {}\n'''.format(cuda.device_count()))
        return s.getvalue()


def _get_runtime_info():
    return _RuntimeInfo()


def print_runtime_info(out=None):
    """Shows Chainer runtime information.

    Runtime information includes:

    - OS platform

    - Chainer version

    - NumPy version

    - CUDA information

    Args:
        out: Output destination.
            If it is ``None``, runtime information
            will be shown in ``sys.stdout``.

    """
    if out is None:
        out = sys.stdout
    out.write(str(_get_runtime_info()))
    if hasattr(out, 'flush'):
        out.flush()
