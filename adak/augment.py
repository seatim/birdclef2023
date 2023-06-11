
"""Functions for data augmentation.
"""

from functools import partial

from fastai.vision.augment import (_draw_mask, affine_mat, AffineCoordTfm,
                                   PadMode)
from torch import stack, zeros_like as t0, ones_like as t1


def htrans_mat(x, p, max_trans):
    """Return a random horizontal translation matrix

    Args:
        x (tensor): an image stack: a tensor of shape (N, C, H, W) where N is
            number of images, C is number of channels, and H and W are height
            and width.

        p (float): probability of returning a non-identity matrix

        max_trans (float): maximum translation in normalized units

    Returns:
        A tensor of shape (N, 3, 3)

    """
    if not (0 <= p <= 1):
        raise ValueError('p must be >= 0 and <= 1')
    if max_trans < 0:
        raise ValueError('max_trans must be >= 0')

    def _def_draw(x):
        return x.new_empty(x.size(0)).uniform_(-max_trans, max_trans)

    mask = _draw_mask(x, _def_draw, draw=None, p=p, batch=False)
    return affine_mat(t1(mask), t0(mask), mask,
                      t0(mask), t1(mask), t0(mask))


class HTrans(AffineCoordTfm):
    """Horizontal translation transform
    """
    def __init__(self,
                 max_trans=0.5,  # Maximum magnitude of translation
                 p=0.75,  # Probability of applying translation
                 mode='bilinear',  # PyTorch `F.grid_sample` interpolation
                 align_corners=True,  # PyTorch `F.grid_sample` align_corners
                 ):
        aff_fs = partial(htrans_mat, p=p, max_trans=max_trans)
        super().__init__(aff_fs, size=None, mode=mode, pad_mode=PadMode.Zeros,
                         align_corners=align_corners, p=p)
