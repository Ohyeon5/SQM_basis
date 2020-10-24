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
    torch.save({
      'conv_module_state_dict': self.conv_module.state_dict(),
      'encoder_module_state_dict': self.encoder_module.state_dict(),
      'decoder_module_state_dict': self.decoder_module.state_dict()
    })

  def load_checkpoint(self, path):
    checkpoint = torch.load(path)
    self.conv_module.load_state_dict(checkpoint['conv_module_state_dict'])
    self.encoder_module.load_state_dict(checkpoint['encoder_module_state_dict'])
    self.decoder_module.load_state_dict(checkpoint['decoder_module_state_dict'])
