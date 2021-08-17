import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig

from models import Wrapper

import os

from SQM_discreteness.hdf5_loader import ToTensor, TimeShuffle

from data_utils import VernierDataModule

from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="lr-vernier-classification", entity="lpsy_sqm", job_type='test')

pl.seed_everything(42) # seed all PRNGs for reproducibility

@hydra.main(config_path='test_conf_1', config_name='config')
def main_func(cfg: DictConfig) -> None:
  print("Testing model {}".format(cfg.model_artifact))

  model_artifact = wandb_logger.experiment.use_artifact(cfg.model_artifact)
  model_path = model_artifact.download()

  model = Wrapper.load_from_checkpoint(os.path.join(model_path, cfg.model_filename))

  trainer = pl.Trainer(gpus=1, logger=wandb_logger)

  test_data_artifact = wandb_logger.experiment.use_artifact(cfg.test_data_artifact)
  test_dataset = test_data_artifact.download()
  test_data_module = VernierDataModule(os.path.join(test_dataset, cfg.test_data_filename), cfg.batch_size, head_n=cfg.head_n, 
  test_data_path=os.path.join(test_dataset, cfg.test_data_filename), ds_transform=[TimeShuffle(), ToTensor(cfg.is_channels_last)])

  trainer.test(model, datamodule=test_data_module)

main_func()