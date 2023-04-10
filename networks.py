import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from utils import default_unet_features
from layers import ResizeTransform, VecInt, SpatialTransformer

class ConvBlock(nn.Module):
  """""""""
  ndims=2, in_channels = 2, out_channels = 16
  ConvBlock(
    (main): Conv3d(2, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (activation): LeakyReLU(negative_slope=0.2)
  )
  """""""""
  def __init__(self, ndims, in_channels, out_channels, stride=1):
    super().__init__()

    Conv = getattr(nn, 'Conv%dd' % ndims)
    self.main = Conv(in_channels, out_channels, 3, stride, 1)
    self.activation = nn.LeakyReLU(0.2)

  def forward(self, x):
    out = self.main(x)
    out = self.activation(out)
    return out

class Unet(nn.Module):
  def __init__(self, inshape=None, nb_features=None, max_pool=2, infeats=None):
    super().__init__()

    ndims = len(inshape)

    if nb_features is None:
      nb_features = default_unet_features()

    enc_nf, dec_nf = nb_features
    nb_dec_convs = len(enc_nf)
    final_convs = dec_nf[nb_dec_convs:]   # [16, 16]
    dec_nf = dec_nf[:nb_dec_convs]        # [32, 32, 32, 32]
    self.nb_levels = nb_dec_convs + 1

    max_pool = [max_pool] * self.nb_levels    # [2, 2, 2, 2, 2]
    MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
    self.pooling = [MaxPooling(s) for s in max_pool]
    self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

    prev_nf = infeats
    encoder_nfs = [prev_nf]
    self.encoder = nn.ModuleList()

    # エンコーダーの構築
    for level in range(self.nb_levels - 1):
      convs = nn.ModuleList()
      nf = enc_nf[level]
      convs.append(ConvBlock(ndims, prev_nf, nf))
      prev_nf = nf
      self.encoder.append(convs)
      encoder_nfs.append(prev_nf)

    # デコーダーの構築
    encoder_nfs = np.flip(encoder_nfs)
    self.decoder = nn.ModuleList()
    for level in range(self.nb_levels - 1):
      convs = nn.ModuleList()
      nf = dec_nf[level]
      convs.append(ConvBlock(ndims, prev_nf, nf))
      prev_nf = nf
      self.decoder.append(convs)
      prev_nf += encoder_nfs[level]

    self.remaining = nn.ModuleList()
    for num, nf in enumerate(final_convs):
      self.remaining.append(ConvBlock(ndims, prev_nf, nf))
      prev_nf = nf

    self.final_nf = prev_nf

  def forward(self, x):
    x_history = [x]

    # エンコーダーの順伝播
    for level, convs in enumerate(self.encoder):
      for conv in convs:
        x = conv(x)
      x_history.append(x)
      x = self.pooling[level](x)

    # デコーダーの順伝播
    for level, convs in enumerate(self.decoder):
      for conv in convs:
        x = conv(x)
        x = self.upsampling[level](x)
        x = torch.cat([x, x_history.pop()], dim=1)

    for conv in self.remaining:
      x = conv(x)

    logging.debug(f"x's construction: {x}")
    return x

class VxmDense(nn.Module):
  def __init__(
    self, 
    inshape, 
    nb_unet_features=None, 
    src_feats=1, 
    trg_feats=1,
    int_downsize=2,
    int_steps=7
  ):
    super().__init__()

    ndims = len(inshape)

    self.unet_model = Unet(
      inshape,
      nb_features=nb_unet_features,
      infeats=(src_feats + trg_feats)
    )

    Conv = getattr(nn, 'Conv%dd' % ndims)
    self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

    self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
    self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    self.resize = ResizeTransform(int_downsize, ndims)
    self.fullsize = ResizeTransform(1 / int_downsize, ndims)

    down_shape = [int(dim / int_downsize) for dim in inshape]
    self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

    self.transformer = SpatialTransformer(inshape)

  def forward(self, source, target, registration=False):

    x = torch.cat([source, target], dim=1)
    x = self.unet_model(x)

    flow_field = self.flow(x)

    pos_flow = flow_field
    if self.resize:
      pos_flow = self.resize(pos_flow)

    preint_flow = pos_flow

    pos_flow = self.integrate(pos_flow)
    pos_flow = self.fullsize(pos_flow)

    y_source = self.transformer(source, pos_flow)
    y_target = None

    if not registration:
      return y_source, preint_flow