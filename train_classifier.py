import numpy as np
import torch
import gc

import os
import os.path
import imageio

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

import wandb

def train_classifier(model, n_epochs, train_dl):
  for epoch in range(n_epochs):
    # The mean loss across mini-batches in the current epoch
    for i, batch in enumerate(train_dl):

      predicted_verniers_copy = model_predictions.detach().clone().cpu()

      batch_labels_copy = batch_labels.detach().clone().cpu().numpy()

      #accuracy = sum(np.argmax(predicted_verniers_copy, axis=1) == batch_labels_copy) / len(batch_labels_copy)

      #print("Accuracy:", accuracy)

      #wandb.log({"loss": loss.item(), "accuracy": accuracy.item(), "video sample": wandb.Video(video_sample)})