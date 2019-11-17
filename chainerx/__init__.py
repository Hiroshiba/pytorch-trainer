_available = False


class ndarray(object):
    """Dummy class for type testing."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError('chainerx is not available.')


def is_available():
    return _available
