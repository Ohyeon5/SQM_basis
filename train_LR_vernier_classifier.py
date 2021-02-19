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

def train_LR_vernier_classifier(model, batches, n_batches, criterion=torch.nn.CrossEntropyLoss(), train_conv=True, train_encoder=False, train_decoder=True, device='cpu'):
  # Freeze specified wrapper modules and select only trainable parameters for optimizer
  trainable_parameters = list()
  trainable_parameters += list(model.parameters())
  #if train_conv:
    #trainable_parameters += list(model.conv_module.parameters())
  #else:
    #for param in model.conv_module.parameters():
      #param.require_grad = False
  #if train_encoder:
    #trainable_parameters += list(model.encoder_module.parameters())
  #else:
    #for param in model.encoder_module.parameters():
      #param.require_grad = False
  #if train_decoder:
    #trainable_parameters += list(model.decoder_module.parameters())
  #else:
    #for param in model.decoder_module.parameters():
      #param.require_grad = False

  #torch.nn.init.normal_(model.decoder_module.classifier[-2].weight, mean=2.0, std=5.0)

  #profile_gpu(detailed=False)

  def get_activation(name):
    def hook(model, input, output):
      print("Name", name)
      #print(input)
      module_input = input[0].detach().cpu().numpy()
      module_output = output.detach().cpu().numpy()
      print("Input -- mean: {}; std: {}".format(np.mean(module_input), np.std(module_input)))
      print("Output -- mean: {}; std: {}".format(np.mean(module_output), np.std(module_output)))
    return hook

  #model.encoder_module.register_forward_hook(get_activation('clstm'))

  # Move model to selected device
  model.to(device)

  # TODO change learning rate, change random seeds, change initialization (try He), check gradients + conv filters
  optimizer = torch.optim.Adam(trainable_parameters)
  #optimizer = torch.optim.SGD(trainable_parameters, lr=1e-5, momentum=0.5)

  loss_history = []

  accuracy_buffer = []

  wandb.watch(model)

  for batch_idx in range(n_batches):
    # The mean loss across mini-batches in the current epoch
    total_loss = 0.0

    batch_number = np.random.randint(0, len(batches))
    batch_frames, batch_labels = batches[batch_number]

    batch_labels = torch.from_numpy(batch_labels).float().to(device)
    images = torch.from_numpy(np.stack(batch_frames)).float() # T x B x H x W x C
    images = images.permute(1, 4, 0, 2, 3) # B x C x T x H x W
    images = images.to(device)

    # Clear the gradients from the previous batch
    optimizer.zero_grad()
    # Compute the model outputs
    predicted_verniers = model(images)
    #print(predicted_verniers.dtype)
    # Compute the loss
    loss = criterion(predicted_verniers, batch_labels)
    # Compute the gradients
    loss.backward()
    # Update the model weights
    optimizer.step()

    # Accumulate the loss
    total_loss += loss.item()

    loss_history.append(loss.item())

    #print("Loss for batch {}: {}; Mean loss: {}".format(batch_idx, loss, total_loss / (batch_idx + 1)))

    #wandb.log({"loss": loss.item()})

    if (batch_idx + 1) % 1 == 0:
      #print("CLSTM activation:", activation['clstm_out'])
      #plt.imshow(activation['clstm_out'][0, 0, :, :])
      #plt.show()
      #print("Predicted verniers: {}".format(predicted_verniers))
      #accuracy = sum(predicted_verniers.to('cpu').detach().numpy().argmax(axis=1)==batch_labels.to('cpu').detach().numpy().argmax(axis=1))/len(batch_labels)
      #accuracy_buffer.append(accuracy)
      #print("Ground truth labels: {}".format(batch_labels))
      #print("Accuracy: {}".format(np.mean(accuracy_buffer[-5:])))
      #save_gif(batch_idx, n_frames, batch_frames, batch_size)
      #print_model_diagnostic(model, loss_history, plot=False)
      #print("Batch ", batch_idx + 1)
      print("Batch ", batch_idx + 1, "; Loss: ", loss.item())
      #plot_grad_flow(model.named_parameters())

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

  #sec_last_ff_layer_weight = model.decoder_module.classifier[1].weight
  #sec_last_ff_layer_grad = sec_last_ff_layer_weight.grad
  #sec_weights = sec_last_ff_layer_weight.detach().cpu().numpy()

  if plot:
    plt.subplot(2, 2, 1)
    plt.hist(weights.flatten())
    #sns.kdeplot(weights.flatten())
    #plt.subplot(2, 2, 2)
    #plt.hist(sec_weights.flatten())
    #sns.kdeplot(sec_weights.flatten())
    #plt.title("Weights KDE")
    #plt.show()

    #sec_gradients = sec_last_ff_layer_grad.detach().cpu().numpy()
    #plt.title("Gradients KDE")
    plt.subplot(2, 2, 3)
    plt.hist(gradients.flatten())
    #sns.kdeplot(gradients.flatten())
    #plt.subplot(2, 2, 4)
    #plt.hist(sec_gradients.flatten())
    #sns.kdeplot(sec_gradients.flatten())
    plt.show()

    #plt.title("Loss history")
    #plt.plot(loss_history)
    #plt.show()

  #print("Last layer of FF classifier weights -- mean: {}; std: {}".format(np.mean(weights), np.std(weights)))
  print("Last layer of weights", weights)
  print("Last layer of gradients: ", gradients)
  #print("Last layer of FF classifier gradients -- mean: {}; std: {}".format(np.mean(gradients), np.std(gradients)))

def plot_grad_flow(named_parameters):
  '''Plots the gradients flowing through different layers in the net during training.
  Can be used for checking for possible gradient vanishing / exploding problems.
  
  Usage: Plug this function in Trainer class after loss.backwards() as 
  "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
  ave_grads = []
  max_grads= []
  layers = []
  for n, p in named_parameters:
      if(p.requires_grad) and ("bias" not in n):
          layers.append(n)
          ave_grads.append(p.grad.abs().mean())
          max_grads.append(p.grad.abs().max())
  plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
  plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
  plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
  plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
  plt.xlim(left=0, right=len(ave_grads))
  plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
  plt.xlabel("Layers")
  plt.ylabel("average gradient")
  plt.title("Gradient flow")
  plt.grid(True)
  plt.legend([Line2D([0], [0], color="c", lw=4),
              Line2D([0], [0], color="b", lw=4),
              Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
  plt.show()