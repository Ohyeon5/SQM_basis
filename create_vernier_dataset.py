import argparse
import h5py
import numpy as np

from dataset import BatchMaker

import wandb

arg_parser = argparse.ArgumentParser(description="Generate a dataset of SQM videos")
arg_parser.add_argument('--file-path', type=str)
arg_parser.add_argument('--n-sequences', type=int)
arg_parser.add_argument('--n-objects', type=int, default=1)
arg_parser.add_argument('--n-frames', type=int, default=10)
arg_parser.add_argument('--scale', type=float, default=1.0)
arg_parser.add_argument('--n-channels', type=int, default=3)

command_line_args = arg_parser.parse_args()

if __name__ == '__main__':
  run = wandb.init(project="lr-vernier-classification", job_type='dataset')

  ds_artifact = wandb.Artifact('vernier_decode_1', type='dataset', metadata=vars(command_line_args))

  # Set up the dataset
  print("Creating a batch maker")

  batch_maker = BatchMaker('decode', command_line_args.n_objects, command_line_args.n_sequences, command_line_args.n_frames, (64*command_line_args.scale, 64*command_line_args.scale, command_line_args.n_channels), None)

  print("Generating batches")

  with h5py.File(command_line_args.file_path, 'w') as hdf_file:
    batches_frames, batches_label = batch_maker.generate_batch()

    for batch_idx in range(command_line_args.n_sequences):
      # Change from channels_last to channels_first
      batch_frames = [np.moveaxis(batch_frame[batch_idx], -1, 0).astype('float32') for batch_frame in batches_frames]
      batch_label = batches_label[batch_idx]

      #print("Batch frame 0", batch_frames[0].shape)
      #print("Batch label", batch_label, batch_label.shape)

      group = hdf_file.create_group("vernier_{}".format(batch_idx))

      group.create_dataset('images', data=batch_frames)
      group.create_dataset('label', data='placeholder') # TODO sort this out
      group.create_dataset('label_id', data=batch_label.astype('int64'))
  
  ds_artifact.add_file(command_line_args.file_path)

  run.log_artifact(ds_artifact)
