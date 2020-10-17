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