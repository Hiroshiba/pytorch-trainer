import torch

from pytorch_trainer.training import extension


class FailOnNonNumber(extension.Extension):
    """Trainer extension to raise RuntimeError if parameters contain NaN or Inf.

    Although parameters including non-number such as NaN and Inf are
    unnecessary in most cases, :class:`~pytorch_trainer.training.Trainer` will continue
    to compute even if the parameters in a given optimizer diverge.
    This extension is aimed to reduce unnecessary computations by throwing
    ``RuntimeError`` if the parameters contain NaN or Inf.
    """

    def __call__(self, trainer):
        models = trainer.updater.get_all_models()
        for name, target in models.items():
            for param in target.parameters():
                if not torch.isfinite(param).all():
                    raise RuntimeError(
                        'Kill the process since parameters in optimizer'
                        ' \'{}\' diverge. R.I.P.'.format(name))
