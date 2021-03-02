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
from torch.utils.data import DataLoader, IterableDataset
from models import Wrapper
from SQM_discreteness.models import Primary_conv3D, ConvLSTM_disc_low, ConvLSTM_disc_high, FF_classifier
from SQM_discreteness.hdf5_loader import HDF5Dataset, ToTensor

import numpy as np

import wandb

import h5py

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger

arg_parser = argparse.ArgumentParser(description='Train a deep learning model for various tasks')
arg_parser.add_argument('--n-epochs', type=int)
arg_parser.add_argument('--batch-size', type=int)
arg_parser.add_argument('--training-data-path', type=str)
arg_parser.add_argument('--wandb-notes', type=str)

command_line_args = arg_parser.parse_args()

do_train_hand_gesture_classifier = False
do_train_LR_vernier_classifier = True
n_epochs = command_line_args.n_epochs

#wandb.init(project="lr-vernier-classification", notes=command_line_args.wandb_notes if command_line_args.wandb_notes else "N/A", config={
  #"num_epochs": command_line_args.n_epochs,
  #"batch_size": command_line_args.batch_size
#})
#config = wandb.config

wandb_logger = WandbLogger(project="lr-vernier-classification")

pl.seed_everything(42) # seed all PRNGs for reproducibility

if (do_train_hand_gesture_classifier):
  print("Training end-to-end for hand gesture classification")
  
  training_dataset = HDF5Dataset(command_line_args.training_data_path, transform=ToTensor())
  #config.update({"dataset_size": len(training_dataset)})
  training_dl = DataLoader(training_dataset, batch_size=command_line_args.batch_size, shuffle=False, drop_last=False)

  # TODO also vary hidden channels
  model = Wrapper(Primary_conv3D(), ConvLSTM_disc_low(1), FF_classifier(256, 2, hidden_channels=64), train_conv=True, train_encoder=True, train_decoder=True)
  trainer = pl.Trainer(gpus=1)

  #model.load_checkpoint("latest_checkpoint.tar")
  trainer.fit(model, training_dl)
  #model.save_checkpoint("latest_checkpoint.tar")

if (do_train_LR_vernier_classifier):
  print("Training end-to-end for L/R vernier classification")

  training_dataset = HDF5Dataset(command_line_args.training_data_path)
  #config.update({"dataset_size": len(training_dataset)})
  training_dl = DataLoader(training_dataset, batch_size=command_line_args.batch_size, shuffle=False, drop_last=False)

  # TODO also vary hidden channels
  #model = Wrapper(Primary_conv3D(), ConvLSTM_disc_low(1), FF_classifier(256, 2, hidden_channels=64), train_conv=True, train_decoder=True)
  model = Wrapper(Primary_conv3D(), ConvLSTM_disc_low(1), FF_classifier(256, 2, hidden_channels=4096), train_conv=True, train_encoder=True, train_decoder=True)
  wandb_logger.watch(model, log_freq=1)
  # Set gpus=-1 to use all available GPUs
  # Set deterministic=True to obtain deterministic behavior for reproducibility
  trainer = pl.Trainer(gpus=1, logger=wandb_logger, log_every_n_steps=4, max_epochs=command_line_args.n_epochs, deterministic=True)

  #model.load_checkpoint("latest_checkpoint.tar", load_conv=False, load_encoder=False, load_decoder=False)
  trainer.fit(model, training_dl)

  input_sample = torch.randn((1, 3, 2, 64, 64))
  model.to_onnx("test_model.onnx", input_sample)
  #model.save_checkpoint("latest_checkpoint_phase2.tar")

#wandb.finish()