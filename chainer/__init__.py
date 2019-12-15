from __future__ import absolute_import

import numpy

from chainer import dataset  # NOQA
from chainer import iterators  # NOQA
from chainer import training  # NOQA


# import class and function
# These functions from backends.cuda are kept for backward compatibility
from chainer._runtime_info import print_runtime_info  # NOQA
from chainer.reporter import DictSummary  # NOQA
from chainer.reporter import get_current_reporter  # NOQA
from chainer.reporter import report  # NOQA
from chainer.reporter import report_scope  # NOQA
from chainer.reporter import Reporter  # NOQA
from chainer.reporter import Summary  # NOQA


from chainer import _environment_check


# Check environment conditions
_environment_check.check()


_array_types = None
_cpu_array_types = None


def _load_array_types():
    # Note: this function may not be protected by GIL because of external
    # calls.
    global _array_types
    global _cpu_array_types
    if _array_types is None:
        array_types = [numpy.ndarray]
        cpu_array_types = [numpy.ndarray]

        array_types = tuple(array_types)
        cpu_array_types = tuple(cpu_array_types)

        _array_types = array_types
        _cpu_array_types = cpu_array_types


def get_array_types():
    _load_array_types()
    return _array_types
