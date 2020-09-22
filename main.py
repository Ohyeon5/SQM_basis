# Main parameters
im_dims     = (64, 64, 3)                                           # input images are 64 by 64 pixels, with 3 color channels (RGB)
decode_crit = 'last_frame'
decode_mode = 'sqm'
n_subjs_sqm = 10
batch_size  = {'recons': 16,      'decode': 16,    'sqm': 16  }     # number of sample sequences in a mini-batch
n_batches   = {'recons': 64,      'decode': 64,    'sqm': 5   }     # number of mini-batches per epoch
n_frames    = {'recons': [8, 13], 'decode': 13,    'sqm': 13  }     
n_epochs    = {'recons': 50,      'decode': 100,   'sqm': None}
n_objs      = {'recons': 10,      'decode': 2,     'sqm': 2   }
noise_lvl   = {'recons': 0.9,     'decode': 1e-5,  'sqm': None}
init_lr     = {'recons': 5e-4,    'decode': 1e-5,  'sqm': None}     # initial learning rate; tune this first
do_best_lr  = {'recons': False,   'decode': False, 'sqm': None}     # whether to run find_best_lr to initialize the learning rate
do_run      = {'recons': True,    'decode': True,  'sqm': True}