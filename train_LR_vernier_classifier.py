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

def profile_gpu(detailed=False):
  # print("Profiling GPU memory usage")
  print("Total GPU memory occupied: {}".format(torch.cuda.memory_allocated()))
  if detailed:
    for obj in gc.get_objects():
      try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
          print(type(obj), obj.size(), obj.element_size() * obj.nelement())
      except:
        pass

def train_LR_vernier_classifier(model, n_epochs, train_dl, criterion=torch.nn.CrossEntropyLoss(), train_conv=True, train_encoder=False, train_decoder=True, device='cpu'):
  # Freeze specified wrapper modules and select only trainable parameters for optimizer
  trainable_parameters = list()
  if train_conv:
    trainable_parameters += list(model.conv_module.parameters())
  else:
    for param in model.conv_module.parameters():
      param.require_grad = False
  if train_encoder:
    trainable_parameters += list(model.encoder_module.parameters())
  else:
    for param in model.encoder_module.parameters():
      param.require_grad = False
  if train_decoder:
    trainable_parameters += list(model.decoder_module.parameters())
  else:
    for param in model.decoder_module.parameters():
      param.require_grad = False

  # Move model to selected device
  model.to(device)

  optimizer = torch.optim.Adam(trainable_parameters)

  wandb.watch(model, log='all')

  for epoch in range(n_epochs):
    # The mean loss across mini-batches in the current epoch
    mean_loss = 0.0
    for i, batch in enumerate(train_dl):
      # Move label ids to selected device
      batch_labels = batch['label_id'].to(device)
      # Stack images and move to selected device
      images = torch.stack(batch['images'], 2).to(device) # B x C x T x H x W
      # Clear the gradients from the previous batch
      optimizer.zero_grad()
      # Compute the model outputs
      model_predictions = model(images)
      # Compute the loss
      loss = criterion(model_predictions, batch_labels)
      # Compute the gradients
      loss.backward()
      # Update the model weights
      optimizer.step()

      # Accumulate the loss
      mean_loss += loss.item()

      predicted_verniers_copy = model_predictions.detach().clone().cpu()
      predicted_vernier_classes = np.zeros_like(predicted_verniers_copy)
      predicted_vernier_classes[predicted_verniers_copy > 0.5] = 1.0

      batch_labels_copy = batch_labels.detach().clone().cpu().numpy()

      accuracy = sum(np.argmax(predicted_vernier_classes, axis=1) == np.argmax(batch_labels_copy, axis=1)) / len(batch_labels)

      print("Accuracy:", accuracy)

      video_sample = images.detach().cpu().transpose(1, 2).numpy()

      wandb.log({"loss": loss.item(), "accuracy": accuracy.item(), "video sample": wandb.Video(video_sample)})

      print("Loss after batch {}: {}".format(i, loss.item()))
    
    mean_loss /= len(train_dl)

    print("Loss after epoch {}: {}".format(epoch, mean_loss))

def save_gif(batch_idx, n_frames, batch_frames, batch_size):
  gif_name        = 'test_output_{}.gif'.format(batch_idx)
  display_frames  = []
  for t in range(n_frames):
    display_frames.append(np.hstack([batch_frames[t][b] for b in range(batch_size)]))
  imageio.mimwrite(gif_name, display_frames, duration=0.1)

def print_model_diagnostic(model, loss_history, plot=True):
  #last_ff_layer_weight = model.decoder_module.classifier[-2].weight
  last_ff_layer_weight = model.network[-2].weight
  last_ff_layer_grad = last_ff_layer_weight.grad
  weights = last_ff_layer_weight.detach().cpu().numpy()
  gradients = last_ff_layer_grad.detach().cpu().numpy()

  if plot:
    plt.subplot(2, 2, 1)
    plt.hist(weights.flatten())
    #plt.title("Weights KDE")
    #plt.show()

    #plt.title("Gradients KDE")
    plt.subplot(2, 2, 3)
    plt.hist(gradients.flatten())
    plt.show()

    #plt.title("Loss history")
    #plt.plot(loss_history)
    #plt.show()

  #print("Last layer of FF classifier weights -- mean: {}; std: {}".format(np.mean(weights), np.std(weights)))
  print("Last layer of weights", weights)
  print("Last layer of gradients: ", gradients)
  #print("Last layer of FF classifier gradients -- mean: {}; std: {}".format(np.mean(gradients), np.std(gradients)))