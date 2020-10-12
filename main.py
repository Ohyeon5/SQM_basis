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
from train_hand_gesture_classifier import train_hand_gesture_classifier
from train_LR_vernier_classifier import train_LR_vernier_classifier

arg_parser = argparse.ArgumentParser(description='Train a deep learning model for various tasks')
arg_parser.add_argument('--n-epochs', type=int)
arg_parser.add_argument('--batch-size', type=int)
arg_parser.add_argument('--batches-per-epoch', type=int)

command_line_args = arg_parser.parse_args()

do_train_hand_gesture_classifier = True
do_train_LR_vernier_classifier = True
model = None # TODO replace model
n_epochs = command_line_args.n_epochs
batch_size = command_line_args.batch_size
batches_per_epoch = command_line_args.batches_per_epoch

if (do_train_hand_gesture_classifier):
  optimizer = torch.optim.Adam(model.parameters())
  train_hand_gesture_classifier(model, optimizer, n_epochs, batch_size, batches_per_epoch)

if (do_train_LR_vernier_classifier):
  optimizer = torch.optim.Adam(model.parameters())
  train_LR_vernier_classifier(model, optimizer, n_epochs, batch_size, batches_per_epoch)