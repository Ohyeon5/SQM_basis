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

wandb_logger = WandbLogger(project="lr-vernier-classification-temp", entity="davethephysicist", job_type='test')

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
  print("Testing model")

  # TODO check if verniers different sides in datasets!
  # TODO introduce tools + put them at the start

  model_artifact = wandb_logger.experiment.use_artifact('model_train_LR_vernier_classifier:latest')
  model_path = model_artifact.download()
  print("Download done!")

  #dataset_artifact = wandb_logger.experiment.use_artifact('videos_sqm_V-AV3:v0')
  #dataset_path = dataset_artifact.download()
  #print("Download done!")

  #test_trainer = pl.Trainer(gpus=1, logger=wandb_logger, deterministic=True)

  #data_module = ExperimentDataModule(os.path.join(dataset_path, 'video_data.hdf5'))

  model = Wrapper.load_from_checkpoint(os.path.join(model_path, 'final_model.ckpt'))

  conditions = ['V-PV{}'.format(n) for n in range(5)] + ['V-AV{}'.format(n) for n in range(5)]

  for condition in conditions:
    batch_maker = BatchMaker('sqm', 1, 1, 13, (64, 64, 3), condition)

    batches_frames, batches_label = batch_maker.generate_batch()

    batches_frames = [torch.from_numpy(np.moveaxis(batch_frames, -1, 1).astype('float32')) for batch_frames in batches_frames]

    #print(batches_frames[0].shape)

    images = torch.stack(batches_frames, 2)

    #print(images.shape)

    # B x C x T x H x W
    pred = model(images)

    print("Pred", pred)

    print("Ground truth", batches_label)

main_func()