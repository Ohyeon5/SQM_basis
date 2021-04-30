# model specifications
import os,sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from SQM_discreteness.convlstm_SreenivasVRao import ConvLSTM

class DiscreteEncoder(nn.Module):
  def __init__(self, window, conv_n_feats, n_convBlocks=2, clstm_hidden=[128, 256], return_all_layers=True, disc_type='simple'):
    super().__init__()
    self.disc_type = disc_type
    self.window = window

    # Two layers of convLSTM
    self.primary_convlstm   = ConvLSTM_block(in_channels=conv_n_feats[n_convBlocks], hidden_channels=clstm_hidden[0], return_all_layers=True)
    self.secondary_convlstm = ConvLSTM_block(in_channels=clstm_hidden[0], hidden_channels=clstm_hidden[1], return_all_layers=return_all_layers)

  def forward(self, x):
    # x is a list of images
    if self.disc_type == 'simple':
      return self.forward_simple(x)
    elif self.disc_type == 'redundant':
      return self.forward_redundant(x)

class EncoderHigh(DiscreteEncoder):
  def __init__(self, window, conv_n_feats, n_convBlocks=2, clstm_hidden=[128, 256], return_all_layers=True, disc_type='simple'):
    super().__init__(window, conv_n_feats, n_convBlocks=n_convBlocks, clstm_hidden=clstm_hidden, return_all_layers=return_all_layers, disc_type=disc_type)

  def forward_simple(self, x):
    x = self.primary_convlstm(x)

    # discrete step: high level - simple - every window frame
    img = x[0][:, slice(self.window-1, None, self.window), :, :, :]

    img = self.secondary_convlstm(img)    # img: 5D tensor => B x T x Filters x H x W

    # Base Network: use the last layer only
    img = img[-1][:,-1,:,:,:].squeeze()

    return img

  def forward_redundant(self, x):
    x = self.primary_convlstm(x)

    # discrete step: high level - redundant - repeat the output of nth frame to have same T
    imgs = []
    for t in range(0, x[-1].shape[1], self.window):
      mm = x[0][:,t,:,:,:].unsqueeze(1).repeat(1,min(self.window, x[-1].shape[1]-t),1,1,1)
      imgs.append(mm)
    img = torch.cat(imgs,1)

    img = self.secondary_convlstm(img)    # img: 5D tensor => B x T x Filters x H x W

    # Base Network: use the last layer only
    img = img[-1][:,-1,:,:,:].squeeze()

    return img

class EncoderLow(DiscreteEncoder):
  def __init__(self, window, conv_n_feats, n_convBlocks=2, clstm_hidden=[128, 256], return_all_layers=True, disc_type='simple'):
    super().__init__(window, conv_n_feats, n_convBlocks=n_convBlocks, clstm_hidden=clstm_hidden, return_all_layers=return_all_layers, disc_type=disc_type)

  def forward_redundant(self,x): 
    # discrete step: input is fed every window frames individually, and only the last output of the primary convlstm is saved
    imgs = []
    for t in range(0, x.shape[1], self.window):
      ind_end = t+self.window if t+self.window<x.shape[1] else None
      mm = self.primary_convlstm(x[:,t:ind_end,:,:,:]) # mm: 5D tensor => B x T x Filters x H x W
      imgs.append(mm[0][:,-1,:,:,:].unsqueeze(1).repeat(1,min(self.window,x.shape[1]-t),1,1,1))
    img = torch.cat(imgs,1) # stacked img: 5D tensor => B x T x C x H x W

    img = self.secondary_convlstm(img)    # img: 5D tensor => B x T x Filters x H x W

    # Base Network: use the last layer only
    img = img[-1][:,-1,:,:,:].squeeze()

    return img

  def forward_simple(self, x):  
    # discrete step: input is fed every window frames individually, and only the last output of the primary convlstm is saved
    imgs = []
    for t in range(0, x.shape[1], self.window):
      ind_end = t+self.window if t+self.window<x.shape[1] else None
      mm = self.primary_convlstm(x[:,t:ind_end,:,:,:]) # mm: 5D tensor => B x T x Filters x H x W
      imgs.append(mm[0][:,-1,:,:,:])
    img = torch.stack(imgs,1) # stacked img: 5D tensor => B x T x C x H x W
    
    img = self.secondary_convlstm(img)    # img: 5D tensor => B x T x Filters x H x W

    # Base Network: use the last layer only
    img = img[-1][:,-1,:,:,:].squeeze()

    return img

# 1) Primary feature extraction conv layer
class Primary_conv3D(nn.Module):
  '''
  Primary feedforward feature extraction convolution layers 
  '''
  def __init__(self, norm_type='bn', conv_n_feats=[3, 32, 64]):
    super().__init__()

    # initial parameter settings
    self.conv_n_feats = conv_n_feats

    # primary convolution blocks for preprocessing and feature extraction
    layers = []
    for ii in range(len(conv_n_feats) - 1): 
      block = Conv3D_Block(self.conv_n_feats[ii],self.conv_n_feats[ii+1],norm_type=norm_type)
      layers.append(block)

    self.primary_conv3D = nn.Sequential(*layers)

  def forward(self, x):  
    # arg: x is a list of images
    
    x = self.primary_conv3D(x)
    x = torch.transpose(x, 2, 1)  # Transpose B x C x T x H x W --> B x T x C x H x W

    return x    

# 2) Primary and Secondary convLSTMs
class ConvLSTM_block(nn.Module):
  """Wrap a ConvLSTM for convenience.
  """
  def __init__(self, in_channels, hidden_channels, kernel_size=(3,3), num_layers=1, return_all_layers=True):
    super(ConvLSTM_block, self).__init__()

    # TODO remove once you've cleaned up the config
    #kernel_size = tuple(kernel_size)

    self.convlstm_block   = ConvLSTM(in_channels=in_channels, hidden_channels=hidden_channels, 
                       kernel_size=kernel_size, num_layers=num_layers, bias=True, 
                       batch_first=True, return_all_layers=return_all_layers)

  def forward(self, x):  
    # arg: x is a 5D tensor => B x T x Filters x H x W
    x, _ = self.convlstm_block(x) 
    return x

# 3) Feedforward classifier
class FF_classifier(nn.Module):
  '''
  Feedforward fully connected classifier
  '''
  def __init__(self, in_channels, n_classes, hidden_channels=None, norm_type=None):
    super(FF_classifier, self).__init__()

    if hidden_channels is None:
      self.hidden_channels = n_classes*5
    else: 
      self.hidden_channels = hidden_channels

    avg_pool_size = (4, 4) # tunable
    self.avg_pool_size = avg_pool_size
    self.in_channels = in_channels

    self.avgpool    = nn.AdaptiveAvgPool2d(avg_pool_size)
    self.norm_layer = define_norm(in_channels, norm_type, dim_mode=2)
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(avg_pool_size[0]*avg_pool_size[1]*in_channels, hidden_channels),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(hidden_channels, n_classes)
    )

  def forward(self, x):  
    # arg: x is a 4D tensor B x C x H x W
    x = self.avgpool(x)
    if self.norm_layer is not None:
      x = self.norm_layer(x)
    x = x.contiguous().view(x.shape[0],-1)

    x = self.classifier(x)

    return x    

# Conv3D block 
class Conv3D_Block(nn.Module):
  ''' 
  use conv3D than multiple Conv2D blocks (for a sake of reducing computational burden)
  INPUT dimension: BxCxTxHxW
  '''
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding=1,norm_type=None):
    # kernel_size, stride, padding should be int scalar value, not tuple nor list
    super(Conv3D_Block,self).__init__()
    # parameters
    self.norm_type = norm_type

    # layers
    self.conv      = nn.Conv3d(in_channels,out_channels,kernel_size=(1,kernel_size,kernel_size),
                   stride=(1,stride,stride),padding=(1,padding,padding))
    self.relu      = nn.ReLU(inplace=True)
    self.maxpool   = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))    
    self.norm_layer= define_norm(out_channels,norm_type,dim_mode=3)

  def forward(self,x):

    #print(x.type())
    x = self.conv(x)
    x = self.relu(x)
    x = self.maxpool(x)
    if self.norm_layer is not None:
      x = self.norm_layer(x)

    return x

##################
#  Aid functions #
################## 

# Define normalization type
def define_norm(n_channel, norm_type, n_group=None, dim_mode=2):
  # define and use different types of normalization steps 
  # Referred to https://pytorch.org/docs/stable/_modules/torch/nn/modules/normalization.html
  if norm_type == 'bn':
    if dim_mode == 2:
      return nn.BatchNorm2d(n_channel)
    elif dim_mode==3:
      return nn.BatchNorm3d(n_channel)
  elif norm_type == 'gn':
    if n_group is None: n_group=2 # default group num is 2
    return nn.GroupNorm(n_group,n_channel)
  elif norm_type == 'in':
    return nn.GroupNorm(n_channel,n_channel)
  elif norm_type == 'ln':
    return nn.GroupNorm(1,n_channel)
  elif norm_type == None:
    return
  else:
    return ValueError('Normalization type - '+norm_type+' is not defined yet')



    