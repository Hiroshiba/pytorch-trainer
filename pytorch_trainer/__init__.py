from __future__ import absolute_import

import numpy

from pytorch_trainer import dataset  # NOQA
from pytorch_trainer import iterators  # NOQA
from pytorch_trainer import training  # NOQA


# import class and function
# These functions from backends.cuda are kept for backward compatibility
from pytorch_trainer._runtime_info import print_runtime_info  # NOQA
from pytorch_trainer.reporter import DictSummary  # NOQA
from pytorch_trainer.reporter import get_current_reporter  # NOQA
from pytorch_trainer.reporter import report  # NOQA
from pytorch_trainer.reporter import report_scope  # NOQA
from pytorch_trainer.reporter import Reporter  # NOQA
from pytorch_trainer.reporter import Summary  # NOQA


from pytorch_trainer import _environment_check


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
