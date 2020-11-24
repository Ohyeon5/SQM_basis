import matplotlib.pyplot as plt
import os
import torch

class Wrapper(torch.nn.Module):
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

  def __init__(self, conv_module, encoder_module, decoder_module):
    super(Wrapper, self).__init__()
    self.conv_module = conv_module
    self.encoder_module = encoder_module
    self.decoder_module = decoder_module

  def forward(self, x):
    x = self.conv_module(x)
    x = self.encoder_module(x)
    x = self.decoder_module(x)

    return x

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