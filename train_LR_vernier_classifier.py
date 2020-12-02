import torch

from numpy import load, loadtxt

def train_LR_vernier_classifier(model, n_epochs):
  batch_array = load("train_vernier_1000.npz")
  batch_labels = loadtxt("train_vernier_1000_labels.csv")

  for epoch in range(n_epochs):
    i = 0
    for array_name in batch_array.keys():
      sequence = batch_array[array_name]
      sequence_label = batch_labels[i]
      i += 1

      print(sequence.shape)
      print(sequence_label)