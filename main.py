"""Model training and testing script

This script allows you to train and test the models on a couple of tasks.

There are two supported tasks:
1) Hand gesture classification: based on a sequence of images representing a hand gesture,
the model labels the sequence with a hand gesture
2) L/R vernier discrimination: based on a sequence of images containing a vernier,
the model labels the sequence as a left or right vernier
"""

import argparse
import torch
from torch.utils.data import DataLoader, IterableDataset, random_split
from models import Wrapper
from SQM_discreteness.models import Primary_conv3D, ConvLSTM_disc_low, ConvLSTM_disc_high, FF_classifier
from SQM_discreteness.hdf5_loader import HDF5Dataset, ToTensor

import numpy as np

import wandb

import h5py

import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
import hydra

from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="lr-vernier-classification", job_type='train')

pl.seed_everything(42) # seed all PRNGs for reproducibility

class VernierDataModule(LightningDataModule):
  def __init__(self, data_path, batch_size, head_n=0, ds_transform=None):
    super().__init__()
    self.data_path = data_path
    self.batch_size = batch_size
    self.dataset = HDF5Dataset(self.data_path, transform=ds_transform)

    if head_n:
      self.dataset = [self.dataset[i] for i in range(head_n)]

    n_train = int(0.8 * len(self.dataset))
    n_val = len(self.dataset) - n_train
    self.train_ds, self.val_ds = random_split(self.dataset, [n_train, n_val])

  def train_dataloader(self):
    train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)
    return train_dl

  def val_dataloader(self):
    val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)
    return val_dl

def train_model(training_data_path, batch_size, n_epochs, arch, do_train, head_n, ds_transform=None):
  data_module = VernierDataModule(training_data_path, batch_size, head_n=head_n, ds_transform=ds_transform)
  # Log dataset statistics # TODO look at this again
  wandb_logger.experiment.config.update({"train_ds_len": len(data_module.train_ds), "val_ds_len": len(data_module.val_ds)})

  model = Wrapper(Primary_conv3D(), ConvLSTM_disc_low(arch.encoder_window),
                  FF_classifier(arch.decoder_inchannels, arch.decoder_n_classes, hidden_channels=arch.decoder_hidden_channels),
                  train_conv=do_train.train_conv, train_encoder=do_train.train_encoder, train_decoder=do_train.train_decoder)
  # Log hyperparameters # TODO fix this ugliness
  wandb_logger.experiment.config.update({"encoder_window": arch.encoder_window, "decoder_inchannels": arch.decoder_inchannels, "decoder_n_classes": arch.decoder_n_classes, "decoder_hidden_channels": arch.decoder_hidden_channels})
  wandb_logger.watch(model, log_freq=1)

  # Set gpus=-1 to use all available GPUs
  # Set deterministic=True to obtain deterministic behavior for reproducibility
  trainer = pl.Trainer(gpus=1, logger=wandb_logger, log_every_n_steps=4, max_epochs=n_epochs, deterministic=True)

  #model.load_checkpoint("latest_checkpoint.tar", load_conv=False, load_encoder=False, load_decoder=False)
  trainer.fit(model, data_module)

  input_sample = torch.randn((1, 3, 2, 64, 64))
  model.to_onnx("test_model.onnx", input_sample)
  #model.save_checkpoint("latest_checkpoint_phase2.tar")

@hydra.main(config_path='conf', config_name='config')
def main_func(cfg: DictConfig) -> None:
  #print(OmegaConf.to_yaml(cfg))

  wandb_logger.experiment.config.update({"num_epochs": cfg.rc.n_epochs, "batch_size": cfg.rc.batch_size})

  if (cfg.rc.task == 'train_hand_gesture_classifier'):
    print("Training end-to-end for hand gesture classification")
    train_model(cfg.rc.training_data_path, cfg.rc.batch_size, cfg.rc.n_epochs, cfg.architecture, cfg.rc.do_train, cfg.head_n, ds_transform=ToTensor())
  elif (cfg.rc.task == 'train_LR_vernier_classifier'):
    print("Training end-to-end for L/R vernier classification")
    train_model(cfg.rc.training_data_path, cfg.rc.batch_size, cfg.rc.n_epochs, cfg.architecture, cfg.rc.do_train, cfg.head_n)

main_func()