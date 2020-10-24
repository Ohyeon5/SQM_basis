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

  def save_checkpoint(self, path):
    """Save a checkpoint of the entire wrapper

    Parameters
    ----------
    path : str
        The path to the checkpoint file, .tar extension
    """
    torch.save({
      'conv_module_state_dict': self.conv_module.state_dict(),
      'encoder_module_state_dict': self.encoder_module.state_dict(),
      'decoder_module_state_dict': self.decoder_module.state_dict()
    })

  def load_checkpoint(self, path, load_conv, load_encoder, load_decoder):
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
