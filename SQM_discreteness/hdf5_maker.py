from utils import *

import numpy as np
import h5py
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm


def resize_images(image_list, im_size):
    """Resize a list of images to a given size.
    
    Parameters
    ----------
    image_list : list
      A list of images to resize, in any format supported by PIL
    im_size : int
      The side length of the resized images
    """
    return_list = []
    for im in image_list:
        img = Image.open(im)
        img = img.resize((im_size, im_size), Image.ANTIALIAS)
        np_img = np.array(img)
        return_list.append(np_img)
    return return_list


def create_image_label_list(img_path, group, im_size, skip, all_labels):
    """
    """ 
    label = all_labels['label'].loc[int(group)]
    image_list = os.listdir(os.path.join(img_path, group))
    if len(image_list) < 24:
        return [], []
    image_list = sorted(image_list[:24:skip])
    images = resize_images([os.path.join(img_path, group, i) for i in image_list], im_size)
    return images, label


def make_hdf5(img_path, im_size, skip, all_labels, desired_labels, fname='data_hdf5.h5'):
    """Make an HDF5 file from a directory of images.
    
    Parameters
    ----------
    img_path : str
      The path of the folder containing the images
    im_size : int
      The side length to give the output images
    skip : int
      The number of images to skip over before adding a new image
    all_labels : DataFrame
      ???
    desired_labels : list
      The labels to be considered
    fname : str
      The name of the HDF5 file to output
    """
    indices = list(all_labels[all_labels['label'].isin(desired_labels)].index)
    with h5py.File(fname, 'w') as hf:
        for group in tqdm(indices):
            group = str(group)
            images, label = create_image_label_list(img_path, group, im_size, skip, all_labels)
            if not images:
                print('{} excluded, because of the short length'.format(group))
                continue
            label_id = desired_labels.index(label)  
            hfgroup = hf.create_group(group)
            hfgroup.create_dataset('images', data=images)
            hfgroup.create_dataset('label', data=label)
            hfgroup.create_dataset('label_id', data=label_id)


if __name__ == "__main__":

    # read config.ini and use the settings
    param = get_configs()

    data_path    = param['data_path']
    img_path     = param['img_path']

    train_labels = pd.read_csv(param['csv_train'], names=['label'], sep=';')
    val_labels   = pd.read_csv(param['csv_val'], names=['label'], sep=';')
    all_labels   = pd.read_csv(param['csv_labels'], sep=';')

    labels       = param['labels']
    fn_postfix   = str(len(labels))

    print('labels are {}, length of {}'.format(labels, fn_postfix))

    train_fn     = data_path + os.sep + 'train_hdf5' + fn_postfix + '.h5'
    val_fn       = data_path + os.sep + 'val_hdf5'   + fn_postfix + '.h5'

    maker_params = {'img_path': img_path, 'im_size': param['im_size'], 'skip': param['skip'], 'desired_labels': labels}
    make_hdf5(all_labels=train_labels, fname=train_fn, **maker_params)
    make_hdf5(all_labels=val_labels,   fname=val_fn,   **maker_params)