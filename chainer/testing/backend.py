import functools

import torch

from chainer.testing import _bundle
from chainer.testing import attr


# TODO(hvy): BackendConfig.__enter__ does not have to modify the current
# device. Change it so that it does not.
class BackendConfig(object):

    _props = [
        ('use_cuda', False),
        ('cuda_device', None),  # 0 by default, if use_cuda=True
    ]

    _device = None

    def __init__(self, params):
        if not isinstance(params, dict):
            raise TypeError('params must be a dict.')
        self._contexts = []

        # Default values
        for k, v in self._props:
            setattr(self, k, v)
        # Specified values
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError('Parameter {} is not defined'.format(k))
            setattr(self, k, v)

        self._check_params()
        self._adjust_params()

    def _check_params(self):
        # Checks consistency of parameters
        pass

    def _adjust_params(self):
        # Adjusts parameters, e.g. fill the default values
        if self.use_cuda:
            if self.cuda_device is None:
                self.cuda_device = 0

    @property
    def device(self):
        if self._device is None:
            if self.use_cuda:
                device = torch.device('cuda', self.cuda_device)
            else:
                device = torch.device('cpu')
            self._device = device
        return self._device

    def __repr__(self):
        lst = []
        for k, _ in self._props:
            lst.append('{}={!r}'.format(k, getattr(self, k)))
        return '<BackendConfig {}>'.format(' '.join(lst))

    def get_func_str(self):
        """Returns a string that can be used in method name"""
        lst = []
        for k, _ in self._props:
            val = getattr(self, k)
            if val is True:
                val = 'true'
            elif val is False:
                val = 'false'
            else:
                val = str(val)
            lst.append('{}_{}'.format(k, val))
        return '__'.join(lst)

    def get_pytest_marks(self):
        marks = []
        if self.use_cuda:
            marks.append(attr.gpu)
            if self.cuda_device >= 1:
                marks.append(attr.multi_gpu(self.cuda_device + 1))

        assert all(callable(_) for _ in marks)
        return marks

    def get_tensor(self, np_array):
        return torch.from_numpy(np_array)


def _test_case_generator(base, method_names, params):
    # Defines the logic to generate test case classes parameterized with
    # backends.

    if method_names is not None:
        def method_generator(base_method):
            if base_method.__name__ in method_names:
                return None
            return base_method

        yield (base.__name__, {}, method_generator)

    for i_param, param in enumerate(params):
        backend_config = BackendConfig(param)
        marks = backend_config.get_pytest_marks()
        cls_name = '{}_{}'.format(base.__name__, backend_config.get_func_str())

        def method_generator(base_method):
            # Generates a wrapped test method

            if (method_names is not None
                    and base_method.__name__ not in method_names):
                return None

            # Bind to a new variable.
            backend_config2 = backend_config

            @functools.wraps(base_method)
            def new_method(self, *args, **kwargs):
                return base_method(self, backend_config2, *args, **kwargs)

            # Apply test marks
            for mark in marks:
                new_method = mark(new_method)

            return new_method

        yield (cls_name, {}, method_generator)


def inject_backend_tests(method_names, params):
    if not (method_names is None or isinstance(method_names, list)):
        raise TypeError('method_names must be either None or a list.')
    if not isinstance(params, list):
        raise TypeError('params must be a list of dicts.')
    if not all(isinstance(d, dict) for d in params):
        raise TypeError('params must be a list of dicts.')

    return _bundle.make_decorator(
        lambda base: _test_case_generator(base, method_names, params))
