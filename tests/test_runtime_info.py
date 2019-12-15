import unittest

import six

from pytorch_trainer import _runtime_info
from pytorch_trainer import testing


class TestRuntimeInfo(unittest.TestCase):

    def test_print_runtime_info(self):
        out = six.StringIO()
        _runtime_info.print_runtime_info(out)
        assert out.getvalue() == str(_runtime_info._get_runtime_info())


testing.run_module(__name__, __file__)
