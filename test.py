import pytorch_lightning as pl

from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import WandbLogger

import os

from models import Wrapper

from dataset import BatchMaker

import hydra
from omegaconf import DictConfig

import torch

import numpy as np

import matplotlib.pyplot as plt

import wandb

from captum.attr import (
  IntegratedGradients
)

wandb_logger = WandbLogger(project="lr-vernier-classification", entity="lpsy_sqm", job_type='test')

pl.seed_everything(42) # seed all PRNGs for reproducibility

class ExperimentDataModule(LightningDataModule):
  def __init__(self, data_path, ds_transform=None):
    super().__init__()
    self.data_path = data_path
    self.dataset = HDF5Dataset(self.data_path, transform=ds_transform)

  def test_dataloader(self):
    test_dl = DataLoader(self.dataset, batch_size=8, shuffle=False, drop_last=False)
    return test_dl

@hydra.main(config_path='test_conf', config_name='config')
def main_func(cfg: DictConfig) -> None:
  print("Testing model {}".format(cfg.model_artifact))

  model_artifact = wandb_logger.experiment.use_artifact(cfg.model_artifact)
  model_path = model_artifact.download()

  #dataset_artifact = wandb_logger.experiment.use_artifact('videos_sqm_V-AV3:v0')
  #dataset_path = dataset_artifact.download()
  #print("Download done!")

  #data_module = ExperimentDataModule(os.path.join(dataset_path, 'video_data.hdf5'))

  model = Wrapper.load_from_checkpoint(os.path.join(model_path, cfg.model_filename))

  pv_conditions = ['V-PV{}'.format(n) for n in range(1, 13)]
  av_conditions = ['V-AV{}'.format(n) for n in range(1, 13)]
  conditions = pv_conditions + av_conditions

  pv_accuracy = []
  av_accuracy = []
  pv_cross_entropy = []
  av_cross_entropy = []

  def ig_attribution():
    test_batch_maker = BatchMaker('sqm', 1, 1, 13, (64, 64, 3), 'V-PV1', random_start_pos=False, random_size=False)
    batches_frames, batches_label = test_batch_maker.generate_batch()
    batches_frames = [torch.from_numpy(np.moveaxis(batch_frames, -1, 1).astype('float32')) for batch_frames in batches_frames]
    images = torch.stack(batches_frames, 2)
    model.eval()
    baseline = torch.zeros_like(images)
    ig = IntegratedGradients(model)
    batches_label = batches_label.tolist()
    attributions = ig.attribute(images, baseline, target=batches_label)
    print('IG attributions:', attributions)
    for frame in range(13):
      display_image = images[0, :, frame, :, :].int()
      display_image = torch.transpose(display_image, 0, 2)
      print(display_image)
      plt.axis('off')
      plt.imshow(display_image)
      plt.show()
      frame_attrib = attributions[0, :, frame, :, :]
      max_attrib = torch.max(frame_attrib)
      min_attrib = torch.min(frame_attrib)
      frame_attrib = (frame_attrib - min_attrib) / (max_attrib - min_attrib)
      frame_attrib = torch.transpose(frame_attrib, 0, 2)
      plt.imshow(frame_attrib)
      #plt.hist(frame_attrib.numpy().flatten())
      plt.show()
    frame_attr = np.sum(attributions.numpy(), axis=(0, 1, 3, 4))
    print("Frame attr", type(frame_attr), frame_attr)

  #ig_attribution()

  def test_batch(batch_maker, log_input=False, n_seq_log=4):
    batches_frames, batches_label = batch_maker.generate_batch()

    batches_frames = [torch.from_numpy(np.moveaxis(batch_frames, -1, 1).astype('float32')) for batch_frames in batches_frames]

    images = torch.stack(batches_frames, 2)

    # B x C x T x H x W
    model_predictions = model(images)

    if log_input:
      # Log the test images
      video_sample = images.detach().cpu()[:n_seq_log].transpose(1, 2).numpy().astype('uint8')
      wandb_logger.experiment.log({'video sample': wandb.Video(video_sample)}) # commit = False

    # If pro-vernier, should be reinforced toward ground truth
    # If anti-vernier, should be reinforced toward opposite of ground truth

    softmaxed = torch.nn.functional.softmax(model_predictions, dim=1)
    softmaxed = softmaxed.detach().numpy()
    prediction_label = np.argmax(softmaxed, axis=1)

    accuracy = sum(prediction_label == batches_label) / len(prediction_label)

    cross_entropy = float(torch.nn.functional.cross_entropy(model_predictions, torch.from_numpy(batches_label).type(torch.LongTensor)))

    return accuracy, cross_entropy

  # Test baseline conditions (only one vernier in each frame)
  baseline_accuracies = []
  baseline_cross_entropies = []
  for frame in range(13):
    baseline_accuracy = 0
    baseline_cross_entropy = 0
    batch_maker = BatchMaker('sqm', 1, cfg.batch_size, 13, (64, 64, 3), 'V{}'.format(frame), random_start_pos=cfg.random_start_pos, random_size=cfg.random_size)
    for batch in range(cfg.n_batches):
      batch_accuracy, batch_cross_entropy = test_batch(batch_maker, log_input=cfg.log_test_data)
      baseline_accuracy += batch_accuracy
      baseline_cross_entropy += batch_cross_entropy
    baseline_accuracy = baseline_accuracy / cfg.n_batches
    baseline_cross_entropy = baseline_cross_entropy / cfg.n_batches

    baseline_accuracies.append(baseline_accuracy)
    baseline_cross_entropies.append(baseline_cross_entropy)

  for condition in pv_conditions:
    condition_accuracy = 0
    condition_cross_entropy = 0
    batch_maker = BatchMaker('sqm', 1, cfg.batch_size, 13, (64, 64, 3), condition, random_start_pos=cfg.random_start_pos, random_size=cfg.random_size)
    for batch in range(cfg.n_batches):
      batch_accuracy, batch_cross_entropy = test_batch(batch_maker, log_input=cfg.log_test_data)
      condition_accuracy += batch_accuracy
      condition_cross_entropy += batch_cross_entropy
    pv_accuracy.append(condition_accuracy / cfg.n_batches)
    pv_cross_entropy.append(condition_cross_entropy / cfg.n_batches)

  for condition in av_conditions:
    condition_accuracy = 0
    condition_cross_entropy = 0
    batch_maker = BatchMaker('sqm', 1, cfg.batch_size, 13, (64, 64, 3), condition, random_start_pos=cfg.random_start_pos, random_size=cfg.random_size)
    for batch in range(cfg.n_batches):
      batch_accuracy, batch_cross_entropy = test_batch(batch_maker, log_input=cfg.log_test_data)
      condition_accuracy += batch_accuracy
      condition_cross_entropy += batch_cross_entropy
    av_accuracy.append(condition_accuracy / cfg.n_batches)
    av_cross_entropy.append(condition_cross_entropy / cfg.n_batches)

  log_michael_plot(pv_accuracy, av_accuracy, baseline_accuracies)
  log_michael_plot_ce(pv_cross_entropy, av_cross_entropy, baseline_cross_entropies)

  display_plot(pv_accuracy, av_accuracy, baseline_accuracies)

def log_michael_plot(pv_accuracy, av_accuracy, baseline_accuracies):
  wandb_logger.experiment.log({"Michael plot": wandb.plot.line_series(
    xs=list(range(1, 13)),
    ys=[pv_accuracy, av_accuracy, baseline_accuracies],
    keys=["Pro-vernier accuracy", "Anti-vernier accuracy", "Baseline accuracy"],
    title="Michael plot",
    xname="Frame number",
  )})

def log_michael_plot_ce(pv_cross_entropy, av_cross_entropy, baseline_cross_entropies):
  wandb_logger.experiment.log({"Michael cross-entropy plot": wandb.plot.line_series(
    xs=list(range(1, 13)),
    ys=[pv_cross_entropy, av_cross_entropy, baseline_cross_entropies],
    keys=["Pro-vernier cross-entropy", "Anti-vernier cross-entropy", "Baseline cross-entropy"],
    title="Michael cross-entropy plot",
    xname="Frame number"
  )})

def display_plot(pv_accuracy, av_accuracy, baseline_accuracies):
  plt.plot(list(range(1, 13)), pv_accuracy, 'r-', label="Pro-vernier accuracy")
  plt.plot(list(range(1, 13)), av_accuracy, 'b-', label="Anti-vernier accuracy")
  #plt.plot(list(range(1, 13)), baseline_accuracies, 'g-', label="Baseline accuracy") TODO fix this
  plt.legend()
  plt.show()

main_func()