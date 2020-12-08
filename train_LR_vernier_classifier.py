import numpy as np
import torch

from dataset import BatchMaker

def train_LR_vernier_classifier(model, n_epochs, epoch_size, batch_size, criterion=torch.nn.CrossEntropyLoss(), train_conv=True, train_encoder=False, train_decoder=True, device='cpu'):
  n_objects = 2
  n_frames = 13
  scale = 2
  n_channels = 3
  batch_maker = BatchMaker('decode', n_objects, batch_size, n_frames, (64*scale, 64*scale, n_channels), None)

  batches_per_epoch = int(epoch_size / batch_size)

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

  for epoch in range(n_epochs):
    # The mean loss across mini-batches in the current epoch
    mean_loss = 0.0
    for batch_idx in range(batches_per_epoch):
      batch_frames, batch_labels = batch_maker.generate_batch()

      batch_labels = torch.from_numpy(batch_labels).long().to(device)

      images = torch.from_numpy(np.stack(batch_frames)).float() # T x B x H x W x C
      images = images.permute(1, 4, 0, 2, 3) # B x C x T x H x W
      images = images.to(device)
      # Clear the gradients from the previous batch
      optimizer.zero_grad()
      # Compute the model outputs
      predicted_verniers = model(images)
      # Compute the loss
      loss = criterion(predicted_verniers, batch_labels)
      # Compute the gradients
      loss.backward()
      # Update the model weights
      optimizer.step()

      # Accumulate the loss
      mean_loss += loss

      print("Loss after batch {}: {}".format(batch_idx, loss))

    mean_loss /= batches_per_epoch

    print("Loss after epoch {}: {}".format(epoch, mean_loss))

    