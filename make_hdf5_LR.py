import h5py

def make_hdf(batch_arrays, file_name):
  hdf_file = h5py.File(file_name, mode='w')
  