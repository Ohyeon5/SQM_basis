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
from models import Wrapper
from SQM_discreteness.hdf5_loader import ToTensor
from SQM_discreteness.models import EncoderHigh

import os

import numpy as np

import wandb

import h5py

import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from dataset import BatchMaker

from data_utils import VernierDataModule

from config_schema import JobConfig, ModelConfig

from SQM_discreteness.models import ConvLSTM_block

from SQM_discreteness.convlstm_SreenivasVRao import ConvLSTM

#config_store = ConfigStore.instance()
#config_store.store(name="config", node=JobConfig)

wandb_logger = WandbLogger(project="lr-vernier-classification", entity="lpsy_sqm", log_model=True, job_type='train')

# TODO log this as well!
#pl.seed_everything(1) # seed all PRNGs for reproducibility

def train_model(model, data_module, n_epochs, val_every_n=4):
  # Log dataset statistics
  wandb_logger.experiment.config.update({"train_ds_len": len(data_module.train_ds), "val_ds_len": len(data_module.val_ds)})
  
  wandb_logger.watch(model, log_freq=1)

  # gradient_clip_val=0.5 for gradient clipping
  trainer = pl.Trainer(gpus=1, logger=wandb_logger, log_every_n_steps=4, min_epochs=16, max_epochs=n_epochs,
    callbacks=[EarlyStopping('loss', patience=10)], deterministic=False, check_val_every_n_epoch=val_every_n)

  trainer.fit(model, data_module)

  #input_sample = torch.randn((1, 3, 2, 64, 64))
  #model.to_onnx("portable_model.onnx", input_sample)

  return trainer

@hydra.main(config_path='train_conf', config_name='config')
def main_func(cfg: DictConfig) -> None:
  #print(OmegaConf.to_yaml(cfg, resolve=True))

  print("Training model {} on task {} for maximum {} epochs".format(cfg.model.arch_id, cfg.rc.task, cfg.rc.n_epochs))

  wandb_logger.experiment.config.update({"num_epochs": cfg.rc.n_epochs, "batch_size": cfg.rc.batch_size})
  # TODO log the cfg.rc dictionary!

  model_identifier = "{}_{}_{}".format(cfg.model.arch_id, cfg.rc.task, cfg.model_uuid)
  model_artifact = wandb.Artifact("model_{}".format(model_identifier), type='model', metadata=OmegaConf.to_container(cfg, resolve=True))

  train_data_artifact = wandb_logger.experiment.use_artifact(cfg.rc.train_data_artifact)
  train_dataset = train_data_artifact.download()

  if cfg.rc.separate_val:
    val_data_artifact = wandb_logger.experiment.use_artifact(cfg.rc.val_data_artifact)
    val_dataset = val_data_artifact.download()
    data_module = VernierDataModule(os.path.join(train_dataset, cfg.rc.train_data_filename), cfg.rc.batch_size, head_n=cfg.head_n,
      val_data_path=os.path.join(val_dataset, cfg.rc.val_data_filename), ds_transform=ToTensor(cfg.rc.is_channels_last))
  else:
    data_module = VernierDataModule(os.path.join(train_dataset, cfg.rc.train_data_filename), cfg.rc.batch_size, head_n=cfg.head_n, ds_transform=ToTensor(cfg.rc.is_channels_last))

  do_train = cfg.rc.do_train

  if cfg.load_model:
    print("Loading model", cfg.model_artifact)
    prev_model_artifact = wandb_logger.experiment.use_artifact(cfg.model_artifact)
    prev_model_path = prev_model_artifact.download()
    model = Wrapper.load_from_checkpoint(os.path.join(prev_model_path, cfg.model_filename),
      train_conv=do_train.train_conv, train_encoder=do_train.train_encoder, train_decoder=do_train.train_decoder)
  else:
    model = Wrapper(cfg.model.conv_module, cfg.model.encoder_module, cfg.model.decoder_module,
      train_conv=do_train.train_conv, train_encoder=do_train.train_encoder, train_decoder=do_train.train_decoder)

  trainer = train_model(model, data_module, cfg.rc.n_epochs, cfg.rc.val_every_n)  

  trainer.save_checkpoint(cfg.model_filename)
  model_artifact.add_file(cfg.model_filename)

  wandb_logger.experiment.log_artifact(model_artifact)

main_func()