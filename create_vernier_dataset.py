import argparse
import numpy as np

from dataset import BatchMaker

arg_parser = argparse.ArgumentParser(description="Generate a dataset of SQM videos")
arg_parser.add_argument('--n-sequences', type=int)
arg_parser.add_argument('--batch-size', type=int)

command_line_args = arg_parser.parse_args()

if __name__ == '__main__':
  # Set up the dataset
  print("Creating a batch maker")

  n_objects = 1
  n_frames = 2
  scale = 1
  n_channels = 3
  batch_maker = BatchMaker('decode', n_objects, command_line_args.batch_size, n_frames, (64*scale, 64*scale, n_channels), None)

  print("Generating batches")

  batches = []
  batches_labels = []
  for batch_idx in range(command_line_args.n_sequences):
    batch_frames, batch_labels = batch_maker.generate_batch()
    batch_labels_opposite = 1 - batch_labels
    batch_labels = np.vstack((batch_labels_opposite, batch_labels)).T

    batches.append(batch_frames)
    batches_labels.append(batch_labels)

  print("Done generating batches")

  with open("vernier_batch.npz", "wb") as outfile:
    np.savez(outfile, *batches)

  with open("vernier_batch_label.npz", "wb") as outfile:
    np.savez(outfile, *batches_labels)