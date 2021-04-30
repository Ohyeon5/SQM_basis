import argparse
import h5py
import numpy as np

from dataset import BatchMaker

from tqdm import tqdm

import wandb

from omegaconf import DictConfig, OmegaConf
import hydra

# TODO remember to use sweeps instead of manually creating every condition

@hydra.main(config_path='ds_conf', config_name='config')
def main_func(cfg: DictConfig) -> None:
  run = wandb.init(project="lr-vernier-classification", job_type='dataset')

  ds_artifact = wandb.Artifact("videos_{}_{}".format(cfg.sequence_type, cfg.condition), type='dataset', metadata=dict(cfg))

  # Set up the dataset
  print("Creating a batch maker")

  batch_maker = BatchMaker(cfg.sequence_type, cfg.n_objects, cfg.n_sequences, cfg.n_frames, (64*cfg.scale, 64*cfg.scale, cfg.n_channels), cfg.condition)

  print("Generating batches (by frame)")

  # TODO change wd with Hydra instead of specifying absolute path
  with h5py.File(hydra.utils.to_absolute_path(cfg.file_path), 'w') as hdf_file:
    batches_frames, batches_label = batch_maker.generate_batch()

    print("Writing batches to file")

    for batch_idx in tqdm(range(cfg.n_sequences)):
      # Change from channels_last to channels_first -> TODO remove for future datasets since ToTensor already performs this
      batch_frames = [np.moveaxis(batch_frame[batch_idx], -1, 0).astype('float32') for batch_frame in batches_frames]
      print(batch_frames[0].shape)
      batch_label = batches_label[batch_idx]

      #print("Batch frame 0", batch_frames[0].shape)
      #print("Batch label", batch_label, batch_label.shape)

      group = hdf_file.create_group("vernier_{}".format(batch_idx))

      group.create_dataset('images', data=batch_frames)

      group.create_dataset('label', data='placeholder') # TODO sort this out
      group.create_dataset('label_id', data=batch_label.astype('int64'))
  
  # TODO change wd with Hydra instead of specifying absolute path
  ds_artifact.add_file(hydra.utils.to_absolute_path(cfg.file_path))

  run.log_artifact(ds_artifact)

if __name__ == '__main__':
  main_func()
