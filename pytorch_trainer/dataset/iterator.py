class Iterator(object):
    """Base class of all dataset iterators.

    Iterator iterates over the dataset, yielding a minibatch at each
    iteration. Minibatch is a list of examples. Each implementation should
    implement an iterator protocol (e.g., the :meth:`__next__` method).

    Note that, even if the iterator supports setting the batch size, it does
    not guarantee that each batch always contains the same number of examples.
    For example, if you let the iterator to stop at the end of the sweep, the
    last batch may contain a fewer number of examples.

    The interface between the iterator and the underlying dataset is not fixed,
    and up to the implementation.

    Each implementation should provide the following attributes (not needed to
    be writable).

    - ``batch_size``: Number of examples within each minibatch.
    - ``epoch``: Number of completed sweeps over the dataset.
    - ``epoch_detail``: Floating point number version of the epoch. For
      example, if the iterator is at the middle of the dataset at the third
      epoch, then this value is 2.5.
    - ``previous_epoch_detail``: The value of ``epoch_detail`` at the previous
      iteration. This value is ``None`` before the first iteration.
    - ``is_new_epoch``: ``True`` if the epoch count was incremented at the last
      update.

    Each implementation should also support serialization to resume/suspend the
    iteration.

    """

    def __del__(self):
        self.finalize()

    def __iter__(self):
        """Returns self."""
        return self

    def __next__(self):
        """Returns the next batch.

        This is a part of the iterator protocol of Python. It may raise the
        :class:`StopIteration` exception when it stops the iteration.

        """
        raise NotImplementedError

    def next(self):
        """Python2 alternative of ``__next__``.

        It calls :meth:`__next__` by default.
        """
        return self.__next__()

    def finalize(self):
        """Finalizes the iterator and possibly releases the resources.

        This method does nothing by default. Implementation may override it to
        better handle the internal resources.

        This method can be called multiple times.

        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finalize()

    def state_dict(self):
        """Serializes the internal state of the iterator.

        .. note::
           It should only serialize the internal state that changes over the
           iteration. It should not serialize what is set manually by
           users such as the batch size.

        """
        pass

    def load_state_dict(self, state_dict):
        """Deserializes the internal state of the iterator.

        .. note::
           It should only serialize the internal state that changes over the
           iteration. It should not serialize what is set manually by
           users such as the batch size.

        """
        pass
