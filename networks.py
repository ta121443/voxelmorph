import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch.distributions.normal import Normal

from utils import default_unet_features

class Unet(nn.Module):
  def __init__(self, inshape=None, nb_features=None, nb_levels=None, nb_conv_per_level=1, max_pool=2):
    super().__init__()

    ndims = len(inshape)
    assert ndims in [1, 2, 3], 'データの次元数は1,2,3のうちいずれかである必要があります。'

    if nb_features is None:
      nb_features = default_unet_features()

    enc_nf, dec_nf = nb_features
    nb_dec_convs = len(enc_nf)
    final_convs = dec_nf[nb_dec_convs:]   # [16, 16]
    dec_nf = dec_nf[:nb_dec_convs]        # [32, 32, 32, 32]
    self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

    if isinstance(max_pool, int):
      max_pool = [max_pool] * self.nb_levels    # [2, 2, 2, 2, 2]

    MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
unet_input_features = 2
x_train = np.arange(0, 102400).reshape(100, 32, 32)
inshape = (*x_train.shape[1:], unet_input_features)
unet = Unet(inshape=inshape)
a = getattr(nn, 'MaxPool4d')
print(a)