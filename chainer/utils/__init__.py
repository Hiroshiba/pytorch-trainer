import contextlib
import shutil
import sys
import tempfile

import six

# The following alias has been moved to chainer/__init__.py in order to break
# circular imports in Python 2.
# from chainer.utils.walker_alias import WalkerAlias


@contextlib.contextmanager
def tempdir(**kwargs):
    # A context manager that defines a lifetime of a temporary directory.
    ignore_errors = kwargs.pop('ignore_errors', False)

    temp_dir = tempfile.mkdtemp(**kwargs)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=ignore_errors)


def _raise_from(exc_type, message, orig_exc):
    # Raises an exception that wraps another exception.
    message = (
        '{}\n\n'
        '(caused by)\n'
        '{}: {}\n'.format(message, type(orig_exc).__name__, orig_exc))
    new_exc = exc_type(message)
    if sys.version_info < (3,):
        six.reraise(exc_type, new_exc, sys.exc_info()[2])
    else:
        six.raise_from(new_exc.with_traceback(orig_exc.__traceback__), None)
