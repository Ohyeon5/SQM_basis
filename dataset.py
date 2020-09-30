class BouncingBallSequenceMaker():
  """Generate a sequence of frames depicting bouncing balls

  Parameters
  ----------
  set_type : {'recons', 'decode', 'sqm'}
      The type of data to generate
  n_objects : int
      The number of objects in a frame sequence
  batch_s : int
      No clue
  n_frames : int
      The length in frames of a frame sequence
  im_dims : (int, int, int)
      The width, height and number of channels of an image
  condition : str
      The type of SQM, can be 'V', 'V-PVn' or 'V-AVn', with n > 0
  
  """

  def __init__(self, set_type, n_objects, batch_s, n_frames, im_dims, condition='V'):
    self.set_type   = set_type
    self.n_objects  = n_objects
    self.n_max_occl = 0
    self.condition  = condition if condition != 'V' else 'V0'
    self.batch_s    = batch_s
    self.n_frames   = n_frames
    self.n_channels = im_dims[-1]                                 # number of channels of each image
    self.scale      = max(im_dims[0], im_dims[1]) / 64
    self.wn_h       = int(im_dims[0] * self.scale)
    self.wn_w       = int(im_dims[1] * self.scale)
    self.gravity    = 0.0
    self.friction   = 0.0
