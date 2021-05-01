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

  #test_trainer = pl.Trainer(gpus=1, logger=wandb_logger, deterministic=True)

  #data_module = ExperimentDataModule(os.path.join(dataset_path, 'video_data.hdf5'))

  model = Wrapper.load_from_checkpoint(os.path.join(model_path, cfg.model_filename))

  pv_conditions = ['V-PV{}'.format(n) for n in range(1, 13)]
  av_conditions = ['V-AV{}'.format(n) for n in range(1, 13)]
  conditions = pv_conditions + av_conditions

  pv_accuracy = []
  av_accuracy = []
  pv_cross_entropy = []
  av_cross_entropy = []

  def test_batch(condition):
    batch_size = cfg.batch_size
    batch_maker = BatchMaker('sqm', 1, batch_size, 13, (64, 64, 3), condition, random_start_pos=cfg.random_start_pos, random_size=cfg.random_size)

    batches_frames, batches_label = batch_maker.generate_batch()

    batches_frames = [torch.from_numpy(np.moveaxis(batch_frames, -1, 1).astype('float32')) for batch_frames in batches_frames]

    images = torch.stack(batches_frames, 2)

    # B x C x T x H x W
    model_predictions = model(images)

    # If pro-vernier, should be reinforced toward ground truth
    # If anti-vernier, should be reinforced toward opposite of ground truth

    softmaxed = torch.nn.functional.softmax(model_predictions, dim=1)
    softmaxed = softmaxed.detach().numpy()
    prediction_label = np.argmax(softmaxed, axis=1)

    accuracy = sum(prediction_label == batches_label) / len(prediction_label)

    cross_entropy = torch.nn.functional.cross_entropy(model_predictions, torch.from_numpy(batches_label).type(torch.LongTensor))

    return accuracy, cross_entropy

  for condition in pv_conditions:
    condition_accuracy, condition_cross_entropy = test_batch(condition)
    pv_accuracy.append(condition_accuracy)
    pv_cross_entropy.append(condition_cross_entropy)

  for condition in av_conditions:
    condition_accuracy, condition_cross_entropy = test_batch(condition)
    av_accuracy.append(condition_accuracy)
    av_cross_entropy.append(condition_cross_entropy)

  log_michael_plot(pv_accuracy, av_accuracy)
  log_michael_plot_ce(pv_cross_entropy, av_cross_entropy)

  display_plot(pv_accuracy, av_accuracy)

def log_michael_plot(pv_accuracy, av_accuracy):
  wandb_logger.experiment.log({"Michael plot": wandb.plot.line_series(
    xs=list(range(1, 13)),
    ys=[pv_accuracy, av_accuracy],
    keys=["Pro-vernier accuracy", "Anti-vernier accuracy"],
    title="Michael plot",
    xname="Frame number",
  )})

def log_michael_plot_ce(pv_cross_entropy, av_cross_entropy):
  wandb_logger.experiment.log({"Michael cross-entropy plot": wandb.plot.line_series(
    xs=list(range(1, 13)),
    ys=[pv_cross_entropy, av_cross_entropy],
    keys=["Pro-vernier cross-entropy", "Anti-vernier cross-entropy"],
    title="Michael cross-entropy plot",
    xname="Frame number"
  )})

def display_plot(pv_accuracy, av_accuracy):
  plt.plot(list(range(1, 13)), pv_accuracy, 'r-', label="Pro-vernier accuracy")
  plt.plot(list(range(1, 13)), av_accuracy, 'b-', label="Anti-vernier accuracy")
  plt.legend()
  plt.show()

main_func()