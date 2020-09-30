import torch
import os

def find_best_lr(wrapper, n_objs, im_dims, batch_size, mode='decode', custom=True, from_scratch=False):
  # Search parameters
  n_samples = 100     # number of learning rates to try
  min_lr    = 1e-7    # minimal learning rate
  max_lr    = 1e-0    # maximal learning rate

  # Training loop utilities
  decay_rate = max_lr / min_lr