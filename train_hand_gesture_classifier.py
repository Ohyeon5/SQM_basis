import torch
from SQM_discreteness.hdf5_loader import ToTensor

def train_hand_gesture_classifier(model, optimizer, n_epochs, train_ds, criterion=torch.nn.CrossEntropyLoss()):
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

  batches_per_epoch = len(train_ds) // 1

  tensor_converter = ToTensor()

  for epoch in range(n_epochs):
    # The mean loss across mini-batches in the current epoch
    mean_loss = 0.0
    for i in range(len(train_ds)):
      sample = tensor_converter(train_ds[i])
      train_images = torch.stack(sample['images'])
      train_images = train_images.transpose(0, 1)
      # C x T x W x H
      # print(train_images.shape)
      train_images = [train_images]
      train_label = sample['label']
      train_label_id = torch.tensor(sample['label_id']).unsqueeze(0)
      # Clear the gradients from the previous batch
      optimizer.zero_grad()
      # Compute the model outputs
      predicted_hand_gestures = model(train_images)
      # print("Predicted hand gestures: {}".format(predicted_hand_gestures))
      # Compute the loss
      loss = criterion(predicted_hand_gestures, train_label_id)
      # Compute the gradients
      loss.backward()
      # Update the model weights
      optimizer.step()
      
      # Accumulate the loss
      mean_loss += loss

      print("Loss after batch {}: {}".format(i, loss))
    
    mean_loss /= batches_per_epoch

    print("Loss after epoch {}: {}".format(epoch, mean_loss))

