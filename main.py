"""Model training and testing script

This script allows you to train and test the models on a couple of tasks.

There are two supported tasks:
1) Hand gesture classification: based on a sequence of images representing a hand gesture,
the model labels the sequence with a hand gesture
2) L/R vernier discrimination: based on a sequence of images containing a vernier,
the model labels the sequence as a left or right vernier
"""

from train_hand_gesture_classifier import train_hand_gesture_classifier
from train_LR_vernier_classifier import train_LR_vernier_classifier

do_train_hand_gesture_classifier = True
do_train_LR_vernier_classifier = True
model = None # TODO replace model
n_epochs = None # TODO replace n_epochs

if (do_train_hand_gesture_classifier):
  train_hand_gesture_classifier(model, n_epochs)

if (do_train_LR_vernier_classifier):
  train_LR_vernier_classifier(model, n_epochs)