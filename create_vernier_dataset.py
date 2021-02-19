import argparse
import h5py
import numpy as np

from dataset import BatchMaker

arg_parser = argparse.ArgumentParser(description="Generate a dataset of SQM videos")
arg_parser.add_argument('--file-path', type=str)
arg_parser.add_argument('--n-sequences', type=int)
arg_parser.add_argument('--n-objects', type=int, default=1)
arg_parser.add_argument('--n-frames', type=int, default=10)
arg_parser.add_argument('--scale', type=float, default=1.0)
arg_parser.add_argument('--n-channels', type=int, default=3)

command_line_args = arg_parser.parse_args()

if __name__ == '__main__':
  # Set up the dataset
  print("Creating a batch maker")

  batch_maker = BatchMaker('decode', command_line_args.n_objects, 1, command_line_args.n_frames, (64*command_line_args.scale, 64*command_line_args.scale, command_line_args.n_channels), None)

  print("Generating batches")

  batches = []
  batches_labels = []
  for batch_idx in range(command_line_args.n_sequences):
    batch_frames, batch_labels = batch_maker.generate_batch()

    batch_frames = np.stack([batch_frames[t] for t in range(len(batch_frames))])
    batch_frames = np.squeeze(batch_frames)

    batch_labels_opposite = 1 - batch_labels
    batch_labels = np.vstack((batch_labels_opposite, batch_labels)).T

    #print(batch_frames.shape)
    #print(batch_labels.shape)

    batches.append(batch_frames)
    batches_labels.append(batch_labels)

  print("Done generating batches")

  with h5py.File(command_line_args.file_path, 'w') as hdf_file:
    frames = np.stack(batches)
    labels = np.stack(batches_labels)
    hdf_file.create_dataset("frames", data=frames)
    hdf_file.create_dataset("labels", data=labels)
