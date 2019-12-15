import warnings

import six

import pytorch_trainer
import torch
from pytorch_trainer.dataset import convert
from pytorch_trainer.dataset import iterator as iterator_module
from pytorch_trainer.training import _updater


class StandardUpdater(_updater.Updater):
    """StandardUpdater(\
iterator, optimizer, converter=convert.concat_examples, device=None, \
loss_func=None, loss_scale=None, auto_new_epoch=True, *, input_device=None)

    Standard implementation of Updater.

    This is the standard implementation of :class:`~pytorch_trainer.training.Updater`.
    It accepts one or more training datasets and one or more optimizers.
    The default update routine assumes that there is only one training dataset
    and one optimizer. Users can override this update routine by inheriting
    this class and overriding the :meth:`update_core` method. Each batch is
    converted to input arrays by :func:`pytorch_trainer.dataset.concat_examples` by
    default, which can also be manually set by ``converter`` argument.

    Args:
        iterator: Dataset iterator for the training dataset. It can also be a
            dictionary that maps strings to iterators.
            If this is just an iterator, then the
            iterator is registered by the name ``'main'``.
        optimizer: Optimizer to update parameters. It can also be a dictionary
            that maps strings to optimizers.
            If this is just an optimizer, then the optimizer is
            registered by the name ``'main'``.
        converter: Converter function to build input arrays. Each batch
            extracted by the main iterator and the ``device`` option are passed
            to this function. :func:`pytorch_trainer.dataset.concat_examples` is used
            by default.
        device(device specifier): Device to which the model is sent.
            If ``None``, the device of the model will stay unchanged.
        loss_func: Loss function. The target link of the main optimizer is used
            by default.
        loss_scale (float): Loss scaling factor. Loss scaling is a usefull
            technique to mitigate vanishing gradient issue that tends to happen
            when low precision data type like float16 is used during training.
            If you set loss scaling factor, gradients of loss values are to be
            multiplied by the factor before backprop starts. The factor is
            propagated to whole gradients in a computational graph along the
            backprop. The gradients of parameters are divided by the factor
            just before the parameters are to be updated.
        auto_new_epoch (bool): If ``True``,
            :meth:`~pytorch_trainer.Optimizer.new_epoch` of the main optimizer is
            automatically called when the ``is_new_epoch`` attribute of the
            main iterator is ``True``.
        input_device (device specifier):
            Device to which the training data is sent.
            If ``input_device`` is omitted, it will match the ``device``
            argument.

    Attributes:
        converter: Converter function.
        loss_func: Loss function. If it is ``None``, the target link of the
                   main optimizer is used instead.
        device: Device to which the model is sent.
        input_device: Device to which the training data is sent.
        iteration: Current number of completed updates.
        auto_new_epoch: If ``True``, :meth:`~pytorch_trainer.Optimizer.new_epoch` is
            automatically called by :meth:`update_core`. In this case, the
            :attr:`~pytorch_trainer.Optimizer.use_auto_new_epoch` attribute of each
            optimizer is also set to ``True``. If :meth:`update_core` is
            overridden, the implementation should correctly call
            :meth:`~pytorch_trainer.Optimizer.new_epoch` of each optimizer.

    """

    def __init__(self, iterator, optimizer, model, converter=convert.concat_examples,
                 device=None, loss_func=None, **kwargs):

        if device is not None:
            device = torch.device(device)

        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if not isinstance(optimizer, dict):
            optimizer = {'main': optimizer}
        self._optimizers = optimizer

        if not isinstance(model, dict):
            model = {'main': model}
        self._models = model

        self.converter = converter
        self.loss_func = loss_func
        self.iteration = 0
        self._device = device

    @property
    def device(self):
        return self._device

    @property
    def epoch(self):
        return self._iterators['main'].epoch

    @property
    def epoch_detail(self):
        return self._iterators['main'].epoch_detail

    @property
    def previous_epoch_detail(self):
        return self._iterators['main'].previous_epoch_detail

    @property
    def is_new_epoch(self):
        return self._iterators['main'].is_new_epoch

    def finalize(self):
        """Finalizes the updater object.

        This method calls the `finalize` method of each iterator that
        this updater has.
        It is called at the end of training loops.

        """
        for iterator in six.itervalues(self._iterators):
            iterator.finalize()

    def get_optimizer(self, name):
        """Gets the optimizer of given name.

        Args:
            name (str): Name of the optimizer.

        Returns:
            ~pytorch_trainer.Optimizer: Corresponding optimizer.

        """
        return self._optimizers[name]

    def get_all_optimizers(self):
        """Gets a dictionary of all optimizers for this updater.

        Returns:
            dict: Dictionary that maps names to optimizers.

        """
        return dict(self._optimizers)

    def get_all_models(self):
        """Gets a dictionary of all models for this updater.

        Returns:
            dict: Dictionary that maps names to models.

        """
        return dict(self._models)

    def get_iterator(self, name):
        """Gets the dataset iterator of given name.

        Args:
            name (str): Name of the dataset iterator.

        Returns:
            ~pytorch_trainer.dataset.Iterator: Corresponding dataset iterator.

        """
        return self._iterators[name]

    def update(self):
        """Updates the parameters of the target model.

        This method implements an update formula for the training task,
        including data loading, forward/backward computations, and actual
        updates of parameters.

        This method is called once at each iteration of the training loop.

        """
        self.update_core()
        self.iteration += 1

    def update_core(self):
        iterator = self._iterators['main']
        batch = iterator.next()
        in_arrays = convert._call_converter(
            self.converter, batch, self.device)

        optimizer = self._optimizers['main']
        model = self._models['main']
        loss_func = self.loss_func or model

        for model in self._models.values():
            model.eval()
        optimizer.zero_grad()

        if isinstance(in_arrays, tuple):
            loss = loss_func(*in_arrays)
        elif isinstance(in_arrays, dict):
            loss = loss_func(**in_arrays)
        else:
            loss = loss_func(in_arrays)

        loss.backward()
        optimizer.step()

    def state_dict(self):
        return {
            'iterators': {
                name: iterator.state_dict()
                for name, iterator in six.iteritems(self._iterators)
            },
            'optimizers': {
                name: optimizer.state_dict()
                for name, optimizer in six.iteritems(self._optimizers)
            },
            'models': {
                name: model.state_dict()
                for name, model in six.iteritems(self._models)
            },
            'iteration': self.iteration
        }

    def load_state_dict(self, state_dict):
        saved_iterators = state_dict['iterators']
        assert len(saved_iterators) == len(self._iterators)
        assert set(saved_iterators.keys()) == set(self._iterators.keys())
        for k in saved_iterators.keys():
            self._iterators[k].load_state_dict(saved_iterators[k])

        saved_optimizers = state_dict['optimizers']
        assert len(saved_optimizers) == len(self._optimizers)
        assert set(saved_optimizers.keys()) == set(self._optimizers.keys())
        for k in saved_optimizers.keys():
            self._optimizers[k].load_state_dict(saved_optimizers[k])

        saved_models = state_dict['models']
        assert len(saved_models) == len(self._models)
        assert set(saved_models.keys()) == set(self._models.keys())
        for k in saved_models.keys():
            self._models[k].load_state_dict(saved_models[k])

        self.iteration = state_dict['iteration']
