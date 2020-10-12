import torch

def train_hand_gesture_classifier(model, optimizer, n_epochs, batch_size, batches_per_epoch, criterion=torch.nn.CrossEntropyLoss):
  """Train the provided model to perform hand gesture classification based on a frame sequence.

  Parameters
  ----------
  model : torch.nn.Module
      The model to train
  optimizer : torch.optim.Optimizer
      The optimizer to use
  n_epochs : int
      The number of epochs to train for
  batch_size : int
      The number of labeled sequences in a mini-batch
  batches_per_epoch : int
      The number of mini-batches in an epoch
  criterion : function that takes two tensors and returns a float, optional
      The loss function to use
  """
  for epoch in range(n_epochs):
    for (inputs, labels) in range(batches_per_epoch):
      # Clear the gradients from the previous batch
      optimizer.zero_grad()
      # Compute the model outputs
      predicted_hand_gestures = model(inputs)
      # Compute the loss
      loss = criterion(predicted_hand_gestures, labels)
      # Compute the gradients
      loss.backward()
      # Update the model weights
      optimizer.step()

