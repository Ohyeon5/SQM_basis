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
from SQM_discreteness.models import Primary_conv3D, ConvLSTM_disc_low, ConvLSTM_disc_high, FF_classifier, Net_continuous, Net_disc_low, Net_disc_high
from SQM_discreteness.hdf5_loader import HDF5Dataset, ToTensor

import os

import numpy as np

import wandb

import h5py

import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
import hydra

from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from dataclasses import dataclass

from dataset import BatchMaker

@dataclass
class LoadConfig:
  load_conv: bool = False
  load_encoder: bool = False
  load_decoder: bool = False

@dataclass
class TrainConfig:
  train_conv: bool =  True
  train_encoder: bool = True
  train_decoder: bool = True

# Include log_model=True to store checkpoints on wandb server
wandb_logger = WandbLogger(project="lr-vernier-classification-temp", entity="davethephysicist", log_model=True, job_type='train')

pl.seed_everything(42) # seed all PRNGs for reproducibility

class VernierDataModule(LightningDataModule):
  def __init__(self, data_path, batch_size, head_n=0, val_data_path = None, test_data_path=None, ds_transform=None):
    super().__init__()
    self.batch_size = batch_size
    self.dataset = HDF5Dataset(data_path, transform=ds_transform)

    if head_n:
      self.dataset = [self.dataset[i] for i in range(head_n)]

    if val_data_path:
      self.train_ds = self.dataset
      self.val_ds = HDF5Dataset(val_data_path, transform=ds_transform)
    else:
      n_train = int(0.8 * len(self.dataset))
      n_val = len(self.dataset) - n_train
      self.train_ds, self.val_ds = random_split(self.dataset, [n_train, n_val])

  def train_dataloader(self):
    train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)
    return train_dl

  def val_dataloader(self):
    val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)
    return val_dl

def train_model(model, data_module, n_epochs):
  # Log dataset statistics
  wandb_logger.experiment.config.update({"train_ds_len": len(data_module.train_ds), "val_ds_len": len(data_module.val_ds)})
  
  wandb_logger.watch(model, log_freq=1)

  trainer = pl.Trainer(gpus=1, logger=wandb_logger, log_every_n_steps=4, min_epochs=16, max_epochs=n_epochs, callbacks=[EarlyStopping('loss', patience=10)], deterministic=True)

  trainer.fit(model, data_module)

  input_sample = torch.randn((1, 3, 2, 64, 64))
  model.to_onnx("portable_model.onnx", input_sample)

  return trainer

@hydra.main(config_path='train_conf', config_name='config')
def main_func(cfg: DictConfig) -> None:
  wandb_logger.experiment.config.update({"num_epochs": cfg.rc.n_epochs, "batch_size": cfg.rc.batch_size})

  do_train = cfg.rc.do_train

  model_identifier = "{}_{}".format(cfg.model.arch_id, cfg.rc.task)
  model_artifact = wandb.Artifact("model_{}".format(model_identifier), type='model', metadata=OmegaConf.to_container(cfg, resolve=True))

  if (cfg.rc.task == 'train_hand_gesture_classifier'):
    print("Training end-to-end for hand gesture classification")
    train_data_artifact = wandb_logger.experiment.use_artifact('hand_gestures_train:v0')
    train_dataset = train_data_artifact.download()
    val_data_artifact = wandb_logger.experiment.use_artifact('hand_gestures_val:v0')
    val_dataset = val_data_artifact.download()
    print("Download done!")
    model = hydra.utils.instantiate(cfg.model.model_init)
    
    #TODO remove later
    cfg.head_n = 0
    data_module = VernierDataModule(os.path.join(train_dataset, 'train_hdf52.h5'), cfg.rc.batch_size, head_n=cfg.head_n, val_data_path=os.path.join(val_dataset, 'val_hdf52.h5'), ds_transform=ToTensor())
    trainer = train_model(model, data_module, cfg.rc.n_epochs)
  elif (cfg.rc.task == 'train_LR_vernier_classifier'):
    print("Training end-to-end for L/R vernier classification")
    if cfg.rc.start_model == 'phase1':
      print("Loading phase I model")
      prev_model_artifact = wandb_logger.experiment.use_artifact('model_train_hand_gesture_classifier:latest')
      prev_model_path = prev_model_artifact.download()
      prev_model = Wrapper.load_from_checkpoint(os.path.join(prev_model_path, 'final_model.ckpt'))
    elif cfg.rc.start_model == 'blank':
      model = Wrapper(cfg.conv_module, {}, cfg.encoder_module, {'window': arch.encoder_window},
                  cfg.decoder_module, {'in_channels': arch.decoder_inchannels, 'n_classes': arch.decoder_n_classes, 'hidden_channels': arch.decoder_hidden_channels},
                  train_conv=do_train.train_conv, train_encoder=do_train.train_encoder, train_decoder=do_train.train_decoder)
    
    raw_data_artifact = wandb_logger.experiment.use_artifact('vernier_decode_1:v0')
    raw_dataset = raw_data_artifact.download()
    print("Download done!")
    data_module = VernierDataModule(os.path.join(raw_dataset, 'vernier_decode_1.hdf5'), cfg.rc.batch_size)
    trainer = train_model(prev_model, data_module, cfg.rc.n_epochs)

  trainer.save_checkpoint("final_model.ckpt")
  model_artifact.add_file("final_model.ckpt")

  wandb_logger.experiment.log_artifact(model_artifact)

main_func()