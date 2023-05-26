
"""Module for making fastai vision Learners from pretrained models, for
transfer learning.
"""

import timm

from fastai.data.transforms import get_c
from fastai.learner import Learner, defaults as learner_defaults
from fastai.optimizer import Adam
from fastai.vision.learner import add_head, TimmBody, _timm_norm, default_split
from fastcore.basics import store_attr


def get_model_config(arch, n_in=3):
    model = timm.create_model(
        arch, pretrained=True, num_classes=0, in_chans=n_in)
    return model.default_cfg


def make_pretrain_learner(pre_learn, dls, metrics, normalize=True,
                          loss_func=None, opt_func=Adam,
                          lr=learner_defaults.lr, cbs=None, path=None,
                          model_dir='models', wd=None, wd_bn_bias=False,
                          train_bn=True, moms=(0.95,0.85,0.95)):
    arch = pre_learn.arch
    pretrained = True
    assert pre_learn.pretrained == True, pre_learn.pretrained
    assert pre_learn.normalize == True, pre_learn.normalize

    n_out = get_c(dls)
    body = pre_learn.model[0]
    assert isinstance(body, TimmBody), type(body)

    # Create new timm model from body of pretrained model by adding a head that
    # will output ``n_out`` features.
    model = add_head(
        body, body.model.num_features, n_out, pool=body.needs_pool)

    if normalize:
        cfg = get_model_config(pre_learn.arch)
        _timm_norm(dls, cfg, True)

    splitter = default_split
    learn = Learner(dls=dls, model=model, loss_func=loss_func,
                    opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
                    metrics=metrics, path=path, model_dir=model_dir, wd=wd,
                    wd_bn_bias=wd_bn_bias, train_bn=train_bn, moms=moms)
    learn.freeze()

    # keep track of args for loggers
    store_attr('arch,normalize,n_out,pretrained', self=learn)
    return learn
