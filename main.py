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
from train_classifier import train_classifier
from models import Wrapper
from SQM_discreteness.models import Primary_conv3D, ConvLSTM_disc_low, ConvLSTM_disc_high, FF_classifier
from SQM_discreteness.hdf5_loader import HDF5Dataset, ToTensor

import numpy as np

import wandb

import h5py

arg_parser = argparse.ArgumentParser(description='Train a deep learning model for various tasks')
arg_parser.add_argument('--n-epochs', type=int)
arg_parser.add_argument('--batch-size', type=int)
arg_parser.add_argument('--training-data-path', type=str)
arg_parser.add_argument('--wandb-notes', type=str)

command_line_args = arg_parser.parse_args()

do_train_hand_gesture_classifier = True
do_train_LR_vernier_classifier = False
# TODO also vary hidden channels
model = Wrapper(Primary_conv3D(), ConvLSTM_disc_low(1), FF_classifier(256, 2, hidden_channels=64))
n_epochs = command_line_args.n_epochs

wandb.init(project="lr-vernier-classification", notes=command_line_args.wandb_notes if command_line_args.wandb_notes else "N/A", config={
  "num_epochs": command_line_args.n_epochs,
  "batch_size": command_line_args.batch_size
})
config = wandb.config

if (do_train_hand_gesture_classifier):
  print("Training end-to-end for hand gesture classification")
  
  training_dataset = HDF5Dataset(command_line_args.training_data_path, transform=ToTensor())
  config.update({"dataset_size": len(training_dataset)})
  training_dl = DataLoader(training_dataset, batch_size=command_line_args.batch_size, shuffle=False, drop_last=False)

  model.load_checkpoint("latest_checkpoint.tar")
  train_classifier(model, config.num_epochs, training_dl, device='cuda')
  #model.save_checkpoint("latest_checkpoint.tar")

if (do_train_LR_vernier_classifier):
  print("Training end-to-end for L/R vernier classification")

  training_dataset = HDF5Dataset(command_line_args.training_data_path)
  config.update({"dataset_size": len(training_dataset)})
  training_dl = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

  model.load_checkpoint("latest_checkpoint.tar", load_conv=False, load_encoder=False, load_decoder=False)
  train_classifier(model, config.num_epochs, training_dl, train_encoder=False, device='cuda')
  model.save_checkpoint("latest_checkpoint_phase2.tar")

wandb.finish()