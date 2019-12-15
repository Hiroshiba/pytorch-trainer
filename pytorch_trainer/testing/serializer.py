import os

import torch

from pytorch_trainer import utils


def save_and_load_pth(src, dst):
    """Saves ``src`` to an PTH file and loads it to ``dst``.

    This is a short cut of :func:`save_and_load` using PTH de/serializers.

    Args:
        src: An object to save.
        dst: An object to load to.

    """
    with utils.tempdir() as tempdir:
        path = os.path.join(tempdir, 'tmp.pth')
        torch.save(src.state_dict(), path)
        dst.load_state_dict(torch.load(path))
