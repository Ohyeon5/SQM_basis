import torch

def train_hand_gesture_classifier(model, n_epochs, train_dl, criterion=torch.nn.CrossEntropyLoss(), train_conv=True, train_encoder=True, train_decoder=True, device='cpu'):
  """Train the provided model to perform hand gesture classification based on a frame sequence.

  Parameters
  ----------
  model : torch.nn.Module
      The model to train
  n_epochs : int
      The number of epochs to train for
  train_dl : torch.utils.data.DataLoader
      The data loader providing the training data
  criterion : function that takes two tensors and returns a float, optional
      The loss function to use
  """

  # Freeze specified wrapper modules and select only trainable parameters for optimizer
  trainable_parameters = list()
  if train_conv:
    for param in model.conv_module.parameters():
      param.require_grad = False
    trainable_parameters += list(model.conv_module.parameters())
  if train_encoder:
    for param in model.encoder_module.parameters():
      param.require_grad = False
    trainable_parameters += list(model.encoder_module.parameters())
  if train_decoder:
    for param in model.decoder_module.parameters():
      param.require_grad = False
    trainable_parameters += list(model.decoder_module.parameters())

  # Move model to selected device
  model.to(device)

  optimizer = torch.optim.Adam(trainable_parameters)

  id_to_label = {0: "Swiping Left", 1: "Swiping Right"}

  for epoch in range(n_epochs):
    # The mean loss across mini-batches in the current epoch
    mean_loss = 0.0
    for i, batch in enumerate(train_dl):
      # Move label ids to selected device
      label_ids = batch['label_id'].to(device)
      # Stack images and move to selected device
      images = torch.stack(batch['images'], 2).to(device) # stacked img: 5D tensor => B x C x T x H x W
      # Clear the gradients from the previous batch
      optimizer.zero_grad()
      # Compute the model outputs
      predicted_hand_gestures = model(images)
      # print("Predicted hand gestures: {}".format(predicted_hand_gestures))
      # Compute the loss
      loss = criterion(predicted_hand_gestures, label_ids)
      # Compute the gradients
      loss.backward()
      # Update the model weights
      optimizer.step()
      
      # Accumulate the loss
      mean_loss += loss

      print("Loss after batch {}: {}".format(i, loss))
    
    mean_loss /= len(train_dl)

    print("Loss after epoch {}: {}".format(epoch, mean_loss))

