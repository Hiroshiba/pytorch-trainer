from pytorch_trainer.testing.array import assert_allclose  # NOQA
from pytorch_trainer.testing.backend import BackendConfig  # NOQA
from pytorch_trainer.testing.backend import inject_backend_tests  # NOQA
from pytorch_trainer.testing.helper import assert_warns  # NOQA
from pytorch_trainer.testing.parameterized import from_pytest_parameterize  # NOQA
from pytorch_trainer.testing.parameterized import parameterize  # NOQA
from pytorch_trainer.testing.parameterized import parameterize_pytest  # NOQA
from pytorch_trainer.testing.parameterized import product  # NOQA
from pytorch_trainer.testing.parameterized import product_dict  # NOQA
from pytorch_trainer.testing.serializer import save_and_load_pth  # NOQA
from pytorch_trainer.testing.training import get_trainer_with_mock_updater  # NOQA


def run_module(name, file):
    """Run current test cases of the file.

    Args:
        name: __name__ attribute of the file.
        file: __file__ attribute of the file.
    """

    if name == '__main__':
        import pytest
        pytest.main([file, '-vvs', '-x', '--pdb'])
