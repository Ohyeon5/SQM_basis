import matplotlib.pyplot as plt
import os
import torch

import pytorch_lightning as pl

import wandb

import numpy as np
import imageio

class Wrapper(pl.LightningModule):
  """Wrap a convolutional module, an encoder and a decoder

  Parameters
  ----------
  conv_module : torch.nn.Module
      The convolutional module
  encoder_module : torch.nn.Module
      The encoder module
  decoder_module : torch.nn.Module
      The decoder module    
  """

  def __init__(self, conv_module, encoder_module, decoder_module, criterion=torch.nn.CrossEntropyLoss(), train_conv=False, train_encoder=False, train_decoder=False):
    super().__init__()
    self.conv_module = conv_module
    self.encoder_module = encoder_module
    self.decoder_module = decoder_module

    self.criterion = criterion

    self.train_conv = train_conv
    self.train_encoder = train_encoder
    self.train_decoder = train_decoder

    self.save_hyperparameters()

    # Metrics
    self.train_acc = pl.metrics.Accuracy()

  def forward(self, x):
    x = self.conv_module(x)
    x = self.encoder_module(x)
    x = self.decoder_module(x)

    return x

  def configure_optimizers(self):
    # Freeze specified wrapper modules and select only trainable parameters for optimizer
    trainable_parameters = list()
    if self.train_conv:
      trainable_parameters += list(self.conv_module.parameters())
    else:
      for param in self.conv_module.parameters():
        param.require_grad = False
    if self.train_encoder:
      trainable_parameters += list(self.encoder_module.parameters())
    else:
      for param in self.encoder_module.parameters():
        param.require_grad = False
    if self.train_decoder:
      trainable_parameters += list(self.decoder_module.parameters())
    else:
      for param in self.decoder_module.parameters():
        param.require_grad = False

    optimizer = torch.optim.Adam(trainable_parameters)

    return optimizer

  def training_step(self, train_batch, batch_idx):
    # Move label ids to selected device
    batch_labels = train_batch['label_id']
    # Stack images and move to selected device
    images = torch.stack(train_batch['images'], 2) # B x C x T x H x W
    # Compute the model outputs
    model_predictions = self.conv_module(images)
    model_predictions = self.encoder_module(model_predictions)
    model_predictions = self.decoder_module(model_predictions)
    # Compute the loss
    loss = self.criterion(model_predictions, batch_labels)

    self.log('loss', loss.item())
    # Log accuracy
    self.train_acc(torch.nn.functional.softmax(model_predictions), batch_labels)
    self.log('accuracy', self.train_acc)

    if batch_idx % 128 == 0:
      video_sample = images.detach().cpu().transpose(1, 2).numpy().astype('uint8')
      self.logger.experiment.log({'video sample': wandb.Video(video_sample)}, commit=False)
    
    return loss

  # TODO add structured saving and loading
  def save_checkpoint(self, path, save_conv = True, save_encoder = True, save_decoder = True):
    """Save a checkpoint of the entire wrapper

    Parameters
    ----------
    path : str
        The path to the checkpoint file, .tar extension
    """
    checkpoint = {}

    if save_conv:
      checkpoint['conv_module_state_dict'] = self.conv_module.state_dict()
    if save_encoder:
      checkpoint['encoder_module_state_dict'] = self.encoder_module.state_dict()
    if save_decoder:
      checkpoint['decoder_module_state_dict'] = self.decoder_module.state_dict()

    torch.save(checkpoint, path)

  def load_checkpoint(self, path, load_conv = True, load_encoder = True, load_decoder = True):
    """Load a checkpoint of the wrapper in modular fashion

    Parameters
    ----------
    path : str
        The path to the checkpoint file, .tar extension
    load_conv : bool
        Whether to load the convolutional module
    load_encoder : bool
        Whether to load the encoder module
    load_decoder : bool
        Whether to load the decoder module
    """
    checkpoint = torch.load(path)
    if load_conv:
      self.conv_module.load_state_dict(checkpoint['conv_module_state_dict'])
    if load_encoder:
      self.encoder_module.load_state_dict(checkpoint['encoder_module_state_dict'])
    if load_decoder:
      self.decoder_module.load_state_dict(checkpoint['decoder_module_state_dict'])

    del checkpoint

  def show_conv_filter_rgb(self, conv_layer, fname, out_channel=0):
    fig, ax = plt.subplots()

    # WARNING: detached tensors still share storage with original tensor
    conv_filter_weights = conv_layer.weight.detach().cpu()

    kernel = conv_filter_weights[out_channel].clone().transpose(0, 1)[0]
    # print("Conv filter: {} of shape {}".format(kernel, kernel.shape))

    kernel = (kernel - torch.min(kernel)) / (torch.max(kernel) - torch.min(kernel))

    ax.imshow(kernel, vmin=0, vmax=1)

    if not os.path.exists("conv_filters"):
      os.makedirs("conv_filters")

    fig.savefig("conv_filters/{}".format(fname))

    plt.close(fig)

  def show_conv_filter(self, conv_layer, fname, in_channel=0, out_channel=0):
    fig, ax = plt.subplots()

    # WARNING: detached tensors still share storage with original tensor
    conv_filter_weights = conv_layer.weight.detach().cpu()
    # print("Shape of first conv layer: {}".format(conv_filter_weights.shape))

    # conv_n_feats[layer_idx + 1], conv_n_feats[layer_idx], 1, kernel_size[1]=3, kernel_size[2]=3 -> Why are we not making use of Conv3D as specified (kernel_size[0] = 1)?
    # print("Shape of very first conv filter: {}".format(conv_filter_weights[0][0].shape))
    kernel = conv_filter_weights[out_channel][in_channel][0]
    # print("Very first conv filter: {}".format(first_kernel))
    # Normalize kernel values
    kernel = (kernel - torch.min(kernel)) / (torch.max(kernel) - torch.min(kernel))
    # print("Very first conv filter normalized: {}".format(first_kernel))

    ax.imshow(kernel, cmap='gray', vmin=0, vmax=1)

    if not os.path.exists("conv_filters"):
      os.makedirs("conv_filters")

    fig.savefig("conv_filters/{}".format(fname))

    plt.close(fig)

  def show_conv_layer_filters(self, layer_idx=0):
    conv_layer = self.conv_module.primary_conv3D[layer_idx].conv

    for out_channel in range(conv_layer.weight.shape[0]):
      self.show_conv_filter_rgb(conv_layer, "conv_out{}".format(out_channel), out_channel=out_channel)
      for in_channel in range(conv_layer.weight.shape[1]):
        self.show_conv_filter(conv_layer, "conv_in{}_out{}".format(in_channel, out_channel), in_channel=in_channel, out_channel=out_channel)

def save_gif(batch_idx, n_frames, batch_frames, batch_size):
  gif_name        = 'test_output_{}.gif'.format(batch_idx)
  display_frames  = []
  for t in range(n_frames):
    display_frames.append(np.hstack([batch_frames[t][b] for b in range(batch_size)]))
  imageio.mimwrite(gif_name, display_frames, duration=0.1)