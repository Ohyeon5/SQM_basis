import torch

def train_hand_gesture_classifier(model, optimizer, n_epochs, train_dl, criterion=torch.nn.CrossEntropyLoss()):
  """Train the provided model to perform hand gesture classification based on a frame sequence.

  Parameters
  ----------
  model : torch.nn.Module
      The model to train
  optimizer : torch.optim.Optimizer
      The optimizer to use
  n_epochs : int
      The number of epochs to train for
  train_dl : torch.utils.data.DataLoader
      The data loader providing the training data
  criterion : function that takes two tensors and returns a float, optional
      The loss function to use
  """

  id_to_label = {0: "Swiping Left", 1: "Swiping Right"}

  for epoch in range(n_epochs):
    # The mean loss across mini-batches in the current epoch
    mean_loss = 0.0
    for i, batch in enumerate(train_dl):
      # print("Batch image shape: {}".format(batch['images'][0].shape))
      # print("Train image shape: {}".format(train_images[0].shape))
      # Clear the gradients from the previous batch
      optimizer.zero_grad()
      # Compute the model outputs
      predicted_hand_gestures = model(batch['images'])
      # print("Predicted hand gestures: {}".format(predicted_hand_gestures))
      # Compute the loss
      loss = criterion(predicted_hand_gestures, batch['label_id'])
      # Compute the gradients
      loss.backward()
      # Update the model weights
      optimizer.step()
      
      # Accumulate the loss
      mean_loss += loss

      print("Loss after batch {}: {}".format(i, loss))
    
    mean_loss /= len(train_dl)

    print("Loss after epoch {}: {}".format(epoch, mean_loss))

