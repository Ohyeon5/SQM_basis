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
from torch.utils.data import DataLoader
from train_hand_gesture_classifier import train_hand_gesture_classifier
from train_LR_vernier_classifier import train_LR_vernier_classifier
from models import Wrapper
from SQM_discreteness.models import Primary_conv3D, ConvLSTM_disc_low, ConvLSTM_disc_high, FF_classifier
from SQM_discreteness.hdf5_loader import HDF5Dataset

arg_parser = argparse.ArgumentParser(description='Train a deep learning model for various tasks')
arg_parser.add_argument('--n-epochs', type=int)
arg_parser.add_argument('--batch-size', type=int)
arg_parser.add_argument('--training-data-path', type=str)

command_line_args = arg_parser.parse_args()

do_train_hand_gesture_classifier = True
do_train_LR_vernier_classifier = False
model = Wrapper(Primary_conv3D(), ConvLSTM_disc_low(4), FF_classifier(256, 2, hidden_channels=10))
n_epochs = command_line_args.n_epochs
batch_size = command_line_args.batch_size
training_dataset = HDF5Dataset(command_line_args.training_data_path)
training_dl = DataLoader(training_dataset, batch_size=batch_size)

if (do_train_hand_gesture_classifier):
  print("Training end-to-end for hand gesture classification")
  optimizer = torch.optim.Adam(model.parameters())
  train_hand_gesture_classifier(model, optimizer, n_epochs, training_dataset)

if (do_train_LR_vernier_classifier):
  optimizer = torch.optim.Adam(model.parameters())
  train_LR_vernier_classifier(model, optimizer, n_epochs, batch_size, batches_per_epoch)