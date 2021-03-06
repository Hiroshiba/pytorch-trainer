from pytorch_trainer.training import extension


def observe_value(observation_key, target_func):
    """Returns a trainer extension to continuously record a value.

    Args:
        observation_key (str): Key of observation to record.
        target_func (function): Function that returns the value to record.
            It must take one argument: :class:~pytorch_trainer.training.Trainer object.
    Returns:
        The extension function.

    This extension is triggered each epoch by default.
    To change this, use the ``trigger`` argument with the
    :meth:`Trainer.extend() <pytorch_trainer.training.Trainer.extend>` method.

    """
    @extension.make_extension(
        trigger=(1, 'epoch'), priority=extension.PRIORITY_WRITER)
    def _observe_value(trainer):
        trainer.observation[observation_key] = target_func(trainer)
    return _observe_value


def observe_lr(optimizer_name='main', observation_key='lr'):
    """Returns a trainer extension to record the learning rate.

    Args:
        optimizer_name (str): Name of optimizer whose learning rate is
            recorded.
        observation_key (str): Key of observation to record.

    Returns:
        The extension function.

    This extension is triggered each epoch by default.
    To change this, use the ``trigger`` argument with the
    :meth:`Trainer.extend() <pytorch_trainer.training.Trainer.extend>` method.

    """
    return observe_value(
        observation_key,
        lambda trainer: trainer.updater.get_optimizer(optimizer_name).param_groups[0]["lr"])
