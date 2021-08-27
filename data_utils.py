from pytorch_lightning import LightningDataModule

from SQM_discreteness.hdf5_loader import HDF5Dataset

from torch.utils.data import DataLoader, IterableDataset, random_split

import h5py

class VernierDataModule(LightningDataModule):
  def __init__(self, data_path, batch_size, head_n=0, val_data_path=None, test_data_path=None, ds_transform=None, num_workers=1):
    super().__init__()
    self.batch_size = batch_size
    self.dataset = HDF5Dataset(data_path, transform=ds_transform)
    self.num_workers = num_workers

    if head_n:
      self.dataset = [self.dataset[i] for i in range(head_n)]

    if val_data_path:
      self.train_ds = self.dataset
      self.val_ds = HDF5Dataset(val_data_path, transform=ds_transform)
    else:
      n_train = int(0.8 * len(self.dataset))
      n_val = len(self.dataset) - n_train
      self.train_ds, self.val_ds = random_split(self.dataset, [n_train, n_val])

    if test_data_path:
      self.test_ds = HDF5Dataset(test_data_path, transform=ds_transform)

  def train_dataloader(self):
    train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)
    return train_dl

  def val_dataloader(self):
    val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)
    return val_dl

  def test_dataloader(self):
    test_dl = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)
    return test_dl