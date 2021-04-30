from dataclasses import dataclass

from omegaconf import DictConfig, MISSING

@dataclass
class TrainConfig:
  train_conv: bool =  True
  train_encoder: bool = True
  train_decoder: bool = True

@dataclass
class ModelConfig:
  conv_module: str = MISSING
  encoder_module: str = MISSING
  decoder_module: str = MISSING

# TODO Remove the three schemas below at some point


@dataclass
class RcConfig:
  n_epochs: int = 100
  batch_size: int =  64
  training_data_path: str = "D:/sqm_data/train_hdf52.h5"
  task: str = 'train_hand_gesture_classifier'
  do_train: TrainConfig = MISSING
  start_model: str = 'blank'

@dataclass
class JobConfig:
  model: ModelConfig = MISSING
  rc: RcConfig = MISSING

  head_n: int = MISSING