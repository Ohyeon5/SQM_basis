import torch

def train_test(model, batch_maker, n_batches, batch_size):
  optimizer = torch.optim.Adam(model.parameters())
  loss_function = torch.nn.MSELoss()

  #batch_data, batch_labels = batch_maker.generate_batch(batch_size)
  #batch_data = torch.from_numpy(batch_data)
  #batch_labels = torch.from_numpy(batch_labels)

  for batch_idx in range(n_batches):
    batch_data, batch_labels = batch_maker.generate_batch(batch_size)
    batch_data = torch.from_numpy(batch_data)
    batch_labels = torch.from_numpy(batch_labels)

    optimizer.zero_grad()
    predictions = model(batch_data)
    #print("Pred", predictions, "Label", batch_labels)
    loss = loss_function(predictions, batch_labels)
    loss.backward()
    optimizer.step()

    print("Loss on batch {}: {}".format(batch_idx, loss))

