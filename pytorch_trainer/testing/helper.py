import contextlib
import sys
import warnings


@contextlib.contextmanager
def assert_warns(expected):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        yield

    # Python 2 does not raise warnings multiple times from the same stack
    # frame.
    if sys.version_info >= (3, 0):
        if not any(isinstance(m.message, expected) for m in w):
            try:
                exc_name = expected.__name__
            except AttributeError:
                exc_name = str(expected)

            raise AssertionError('%s not triggerred' % exc_name)
