import argparse
import h5py
import numpy as np

from dataset import BatchMaker

arg_parser = argparse.ArgumentParser(description="Generate a dataset of SQM videos")
arg_parser.add_argument('--n-sequences', type=int)

command_line_args = arg_parser.parse_args()

if __name__ == '__main__':
  # Set up the dataset
  print("Creating a batch maker")

  n_objects = 1
  n_frames = 2
  scale = 1
  n_channels = 3
  batch_maker = BatchMaker('decode', n_objects, 1, n_frames, (64*scale, 64*scale, n_channels), None)

  print("Generating batches")

  batches = []
  batches_labels = []
  for batch_idx in range(command_line_args.n_sequences):
    batch_frames, batch_labels = batch_maker.generate_batch()

    batch_frames = np.stack([batch_frames[t] for t in range(n_frames)])
    batch_frames = np.squeeze(batch_frames)

    batch_labels_opposite = 1 - batch_labels
    batch_labels = np.vstack((batch_labels_opposite, batch_labels)).T

    #print(batch_frames.shape)
    #print(batch_labels.shape)

    batches.append(batch_frames)
    batches_labels.append(batch_labels)

  print("Done generating batches")

  with h5py.File("vernier_data.hdf5", 'w') as hdf_file:
    frames = np.stack(batches)
    labels = np.stack(batches_labels)
    hdf_file.create_dataset("frames", data=frames)
    hdf_file.create_dataset("labels", data=labels)
