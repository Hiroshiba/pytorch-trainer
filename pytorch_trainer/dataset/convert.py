import collections

import numpy
import six
import torch
from torch._six import container_abcs

import pytorch_trainer


class Converter(object):
    """Base class of converters.

    Converters receive batched data retrieved from iterators and perform
    arbitrary transforms as well as device transfer.

    Implementation should override the ``__call__`` method.

    .. seealso::
        :meth:`pytorch_trainer.dataset.converter` --- a decorator to turn a converter
        function into a ``Converter`` instance.

    """

    def __call__(self, batch, device):
        """Performs conversion.

        Args:
            batch:
                A batch. The type and value are arbitrary, depending on usage.
            device(~pytorch_trainer.backend.Device):
                Device to which the converter is expected to send the batch.

        Returns: A converted batch.
        """
        raise NotImplementedError(
            'Concrete class must implement __call__.')


class _ArbitraryCallableConverter(Converter):
    """Converter to wrap a callable with arbitrary arguments.

    This class accepts arbitrary arguments and pass-through to the underlying
    callable, with device argument replaced.
    """

    def __init__(self, base_callable):
        if not callable(base_callable):
            raise TypeError(
                'Can only wrap a callable. Actual: {}'.format(
                    type(base_callable)))
        self.base_callable = base_callable

    def __call__(self, *args, **kwargs):
        base_callable = self.base_callable

        # Normalize the 'device' argument
        if len(args) >= 2:
            # specified as a positional argument
            args = list(args)
            args[1] = _get_device(args[1])
        elif 'device' in kwargs:
            kwargs['device'] = _get_device(kwargs['device'])
        return base_callable(*args, **kwargs)


def converter():
    """Decorator to make a converter.

    This decorator turns a converter function into a
    :class:`pytorch_trainer.dataset.Converter` class instance, which also is a
    callable.
    This is required to use the converter function from an old module that
    does not support :class:`pytorch_trainer.backend.Device` instances
    (See the **Device argument conversion** section below).

    .. rubric:: Requirements of the target function

    The target converter function must accept two positional arguments:
    a batch and a device, and return a converted batch.

    The type of the device argument is :class:`pytorch_trainer.backend.Device`.

    The types and values of the batches (the first argument and the return
    value) are not specified: they depend on how the converter is used (e.g.
    by updaters).

    .. admonition:: Example

        >>> @chainer.dataset.converter()
        ... def custom_converter(batch, device):
        ...     assert isinstance(device, chainer.backend.Device)
        ...     # do something with batch...
        ...     return device.send(batch)

    .. rubric:: Device argument conversion

    For backward compatibility, the decorator wraps
    the function so that if the converter is called with the device argument
    with ``int`` type, it is converted to a :class:`pytorch_trainer.backend.Device`
    instance before calling the original function. The ``int`` value indicates
    the CUDA device of the cupy backend.

    Without the decorator, the converter cannot support ChainerX devices.
    If the batch were requested to be converted to ChainerX with such
    converters, :class:`RuntimeError` will be raised.

    """

    def wrap(func):
        return _ArbitraryCallableConverter(func)

    return wrap


def _call_converter(converter, batch, device):
    # Calls the converter.
    # Converter can be either new-style (accepts pytorch_trainer.backend.Device) or
    # old-style (accepts int as device).
    assert device is None or isinstance(device, torch.device)

    if isinstance(converter, Converter):
        # New-style converter
        return converter(batch, device)

    # Old-style converter
    if device is None:
        return converter(batch, None)
    else:
        return converter(batch, device)


def to_device(device, x):
    """Send an array to a given device.

    This method sends a given array to a given device. This method is used in
    :func:`~pytorch_trainer.dataset.concat_examples`.
    You can also use this method in a custom converter method used in
    :class:`~pytorch_trainer.training.Updater` and :class:`~pytorch_trainer.training.Extension`
    such as :class:`~pytorch_trainer.training.updaters.StandardUpdater` and
    :class:`~pytorch_trainer.training.extensions.Evaluator`.

    See also :func:`pytorch_trainer.dataset.concat_examples`.

    Args:
        device (None or int or device specifier): A device to which an array
            is sent. If it is a negative integer, an array is sent to CPU.
            If it is a positive integer, an array is sent to GPU with the
            given ID. If it is``None``, an array is left in the original
            device. Also, any of device specifiers described at
            :class:`~pytorch_trainer.backend.DeviceId` is accepted.
        x (:ref:`ndarray`): An array to send.

    Returns:
        Converted array.

    """
    device = _get_device(device)

    if device is None:
        return x
    return torch.as_tensor(x).to(device)


def _get_device(device_spec):
    # Converts device specificer to a pytorch_trainer.Device instance.
    # Additionally to pytorch_trainer.get_device, this function supports None
    if device_spec is None:
        return None
    return torch.device(device_spec)


# TODO(hvy): Write unit tests where batch elements contain Python lists.
@converter()
def concat_examples(batch, device=None, padding=None):
    """Concatenates a list of examples into array(s).

    This function converts an "array of tuples" into a "tuple of arrays".
    Specifically, given a list of examples each of which consists of
    a list of elements, this function first makes an array
    by taking the element in the same position from each example
    and concatenates them along the newly-inserted first axis
    (called `batch dimension`) into one array.
    It repeats this for all positions and returns the resulting arrays.

    The output type depends on the type of examples in ``batch``.
    For instance, consider each example consists of two arrays ``(x, y)``.
    Then, this function concatenates ``x`` 's into one array, and ``y`` 's
    into another array, and returns a tuple of these two arrays. Another
    example: consider each example is a dictionary of two entries whose keys
    are ``'x'`` and ``'y'``, respectively, and values are arrays. Then, this
    function concatenates ``x`` 's into one array, and ``y`` 's into another
    array, and returns a dictionary with two entries ``x`` and ``y`` whose
    values are the concatenated arrays.

    When the arrays to concatenate have different shapes, the behavior depends
    on the ``padding`` value. If ``padding`` is ``None`` (default), it raises
    an error. Otherwise, it builds an array of the minimum shape that the
    contents of all arrays can be substituted to. The padding value is then
    used to the extra elements of the resulting arrays.

    .. admonition:: Example

       >>> import numpy as np
       >>> from chainer import dataset
       >>> x = [([1, 2], 1),
       ...      ([3, 4], 2),
       ...      ([5, 6], 3)]
       >>> dataset.concat_examples(x)
       (array([[1, 2],
              [3, 4],
              [5, 6]]), array([1, 2, 3]))
       >>>
       >>> y = [(np.array([1, 2]), 0),
       ...      (np.array([3]), 1),
       ...      (np.array([]), 2)]
       >>> dataset.concat_examples(y, padding=100)
       (array([[  1,   2],
              [  3, 100],
              [100, 100]]), array([0, 1, 2]))
       >>>
       >>> z = [(np.array([1, 2]), np.array([0])),
       ...      (np.array([3]), np.array([])),
       ...      (np.array([]), np.array([2]))]
       >>> dataset.concat_examples(z, padding=(100, 200))
       (array([[  1,   2],
              [  3, 100],
              [100, 100]]), array([[  0],
              [200],
              [  2]]))
       >>> w = [{'feature': np.array([1, 2]), 'label': 0},
       ...      {'feature': np.array([3, 4]), 'label': 1},
       ...      {'feature': np.array([5, 6]), 'label': 2}]
       >>> dataset.concat_examples(w)  # doctest: +SKIP
       {'feature': array([[1, 2],
              [3, 4],
              [5, 6]]), 'label': array([0, 1, 2])}

    Args:
        batch (list): A list of examples. This is typically given by a dataset
            iterator.
        device (device specifier): A device to which each array is sent.
            If it is omitted, all arrays are left in their original devices.
            See :meth:`~pytorch_trainer.dataset.convert.to_device` for more details.
        padding: Scalar value for extra elements. If this is None (default),
            an error is raised on shape mismatch. Otherwise, an array of
            minimum dimensionalities that can accommodate all arrays is
            created, and elements outside of the examples are padded by this
            value.

    Returns:
        Array, a tuple of arrays, or a dictionary of arrays. The type depends
        on the type of each example in the batch.

    """
    assert device is None or isinstance(device, torch.device)
    if not batch:
        raise ValueError('batch is empty')

    first_elem = batch[0]

    if isinstance(first_elem, container_abcs.Sequence):
        result = []
        if not isinstance(padding, tuple):
            padding = [padding] * len(first_elem)

        for i in six.moves.range(len(first_elem)):
            result.append(to_device(device, _concat_arrays(
                [example[i] for example in batch], padding[i])))

        return tuple(result)

    elif isinstance(first_elem, container_abcs.Mapping):
        result = {}
        if not isinstance(padding, dict):
            padding = {key: padding for key in first_elem}

        for key in first_elem:
            result[key] = to_device(device, _concat_arrays(
                [example[key] for example in batch], padding[key]))

        return result

    else:
        return to_device(device, _concat_arrays(batch, padding))


def _concat_arrays(arrays, padding):
    # Convert `arrays` to numpy.ndarray if `arrays` consists of the built-in
    # types such as int, float or list.
    if not isinstance(arrays[0], pytorch_trainer.get_array_types() + (torch.Tensor,)):
        arrays = torch.as_tensor(numpy.asarray(arrays))

    if padding is not None:
        arr_concat = _concat_arrays_with_padding(arrays, padding)
    else:
        arr_concat = torch.stack(
            [array for array in arrays])

    return arr_concat


def _concat_arrays_with_padding(arrays, padding):
    shape = numpy.array(arrays[0].shape, dtype=int)
    for array in arrays[1:]:
        if numpy.any(shape != array.shape):
            numpy.maximum(shape, array.shape, shape)
    shape = tuple(numpy.insert(shape, 0, len(arrays)))

    result = torch.full(shape, padding, dtype=arrays[0].dtype)
    for i in six.moves.range(len(arrays)):
        src = arrays[i]
        slices = tuple(slice(dim) for dim in src.shape)
        result[(i,) + slices] = src

    return result
