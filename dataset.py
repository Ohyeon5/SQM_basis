# Dataset funcions
import numpy as np
from numpy.random import RandomState as rng
from skimage.draw import circle, ellipse, rectangle, polygon
# from skimage.morphology import square, rectangle, diamond, disk, octagon, star
from skimage.transform import resize, rotate

from tqdm import tqdm

import re

import random

# TODO: document fact that c is the number of color channels

class NeilBase():
  def __init__(self, objects, batch_s, scl, n_frames, wn_w, wn_h, grav, random_start_pos, random_start_speed, random_size):
    """
    Parameters
    ----------
    objects: int
    batch_s: int
    scl: float
    n_frames: int
    wn_w: int
    wn_h: int
    """
    self.batch_s = batch_s

    # vx and vy denote the starting velocity of the vernier
    if random_start_speed:
      self.vx         = rng().uniform(-3*scl, 3*scl,   (1, self.batch_s))
      self.vy         = rng().uniform(-3*scl, 3*scl,   (1, batch_s))
    else:
      flow       = (len(objects)%2 - 0.5)*4*scl**2
      self.vx         = np.ones((1, self.batch_s))*flow
      self.vy         = np.ones((1, self.batch_s))*0.0

    # sizx and sizy denote the size of half a vernier
    if random_size:
      self.sizx  = rng().uniform(wn_w/10, wn_w/2, (1, self.batch_s))  # max: /4
      self.sizy  = rng().uniform(wn_h/10, wn_h/2, (1, self.batch_s))  # max: /4
    else:
      self.sizx  = np.ones((1, self.batch_s))*wn_w/5
      self.sizy  = np.ones((1, self.batch_s))*wn_w/4

    # x and y denote the starting position of the center of the vernier
    if random_start_pos:
      x_margin = n_frames * abs(self.vx) + self.sizx
      y_margin = self.sizy + 4
      self.x = rng().uniform(x_margin, wn_w - x_margin, (1, self.batch_s))
      self.y = rng().uniform(y_margin, wn_h - y_margin, (1, self.batch_s))
    else:
      self.x = np.ones((1, self.batch_s)) * wn_w//2
      self.y = np.ones((1, self.batch_s)) * wn_h//2

  # Draw the object (square patch)
  def draw(self, wn, batch_s):
    for b in range(batch_s):
      if self.popped[:, b]:
        patch  = self.patches[b][self.side_[0, b]]/255
        start  = [self.pos[1, b] - patch.shape[0]//2, self.pos[0, b] - patch.shape[1]//2]
        rr, cc = rectangle(start=start, extent=patch.shape, shape=wn.shape[1:3])
        rr     = rr.astype(int)
        cc     = cc.astype(int)
        pat_rr = (rr - self.pos[1, b] - patch.shape[0]/2).astype(int)
        pat_cc = (cc - self.pos[0, b] - patch.shape[1]/2).astype(int)
        bckgrd = wn[b, rr, cc, :]
        for i, color in enumerate(self.colr[:, b]):
          col_patch = color*patch[pat_rr, pat_cc] - bckgrd[:,:,i]
          wn[b, rr, cc, i] += col_patch.clip(0, 255).astype(np.uint8)

  # Update objects position and velocity (of visible objects)
  def update_states(self, batch_s, friction):
    self.vel[:, self.popped[0]] += self.acc[:, self.popped[0]] - self.vel[:, self.popped[0]]*friction
    self.pos[:, self.popped[0]] += self.vel[:, self.popped[0]]

class NeilRecons(NeilBase):
  def __init__(self, set_type, objects, batch_s, scl, n_frames, c, wn_w, wn_h, grav, random_start_pos=False, random_start_speed=False, random_size=False):
    super().__init__(objects, batch_s, scl, n_frames, wn_w, wn_h, grav, random_start_pos, random_start_speed, random_size)

    # Select object static and dynamic properties
    choices    = ['rectangle', 'ellipse', 'vernier']
    self.ori   = rng().uniform(0, 2*np.pi,      (1, batch_s))
    self.colr  = rng().randint(100, 255,        (c, batch_s))
    self.pop_t = rng().randint(0, n_frames//2,  (1, batch_s))

    self.shape  = rng().choice(choices, (1, batch_s))
    self.side   = rng().randint(0, 2, (1, batch_s)) if len(objects) == 0 else objects[0].side
    self.side_  = 1*self.side                   # evolving value for sqm (deep copy)
    self.popped = np.array([[False]*batch_s])   # display stimulus or not
    self.sizx[self.shape == 'vernier'] /= 1.5   # verniers look better if not too wide
    self.sizy[self.shape == 'vernier'] *= 2.0   # verniers appear smaller than other shapes
    self.pos    = np.vstack((self.x,   self.y))
    self.vel    = np.vstack((self.vx, self.vy))
    self.acc    = np.array([[0.00]*batch_s, [grav]*batch_s])

    # Generate patches to draw the shapes efficiently
    self.patches = []
    for b in range(batch_s):
      patch_info = self.generate_patch_info(self.shape[0, b], self.sizx[0, b], self.sizy[0, b], self.ori[0, b])
      self.patches.append(patch_info)

  def generate_patch_info(self, shape_type, sizx, sizy, ori):
    max_s   = int(2*max(sizx, sizy))
    patch   = np.zeros((max_s, max_s))
    patch_0 = None

    if shape_type == 'ellipse':
      center = (patch.shape[0]//2, patch.shape[1]//2)
      radius = (sizy/2, sizx/2) 
      rr, cc = ellipse(center[0], center[1], radius[0], radius[1], shape=patch.shape)
      patch[rr, cc] = 255
    elif shape_type == 'rectangle':
      start  = (int(max_s - sizy//2, int(max_s - sizx)//2))
      extent = (int(sizy), int(sizx))
      rr, cc = rectangle(start=start, extent=extent, shape=patch.shape)
      patch[rr, cc] = 255
    if shape_type == 'vernier':
      patch_0 = np.zeros((max_s, max_s))  # patch with zero offset
      
      v_siz_w = rng().uniform(1 + sizx//6, 1 + sizx//2)
      v_siz_h = rng().uniform(1 + sizy//4, 1 + sizy//2)
      v_off_w = rng().uniform(1,              1 + (sizx - v_siz_w)//2)*2
      v_off_h = rng().uniform(1 + v_siz_h//2, 1 + (sizy - v_siz_h)//2)*2

      start1     = (int((max_s - v_off_h - v_siz_h)//2), int((max_s - v_off_w - v_siz_w)//2))
      start2     = (int((max_s + v_off_h - v_siz_h)//2), int((max_s + v_off_w - v_siz_w)//2))
      start01    = (int((max_s - v_off_h - v_siz_h)//2), int((max_s - 0       - v_siz_w)//2))
      start02    = (int((max_s + v_off_h - v_siz_h)//2), int((max_s + 0       - v_siz_w)//2))
      extent     = (int(v_siz_h), int(v_siz_w))
      rr1,  cc1  = rectangle(start=start1,  extent=extent, shape=patch.shape)
      rr2,  cc2  = rectangle(start=start2,  extent=extent, shape=patch.shape)
      rr01, cc01 = rectangle(start=start01, extent=extent, shape=patch.shape)
      rr02, cc02 = rectangle(start=start02, extent=extent, shape=patch.shape)
      patch[  rr1,  cc1 ] = 255
      patch[  rr2,  cc2 ] = 255
      patch_0[rr01, cc01] = 255
      patch_0[rr02, cc02] = 255

    patch  = rotate(patch, ori).astype(int)
    patch_info = [patch, np.fliplr(patch), patch_0]

    return patch_info

    # Compute what must be updated between the frames
  def compute_changes(self, t, batch_s, objects, cond):
    # Visible objects appear
    self.popped[t >= self.pop_t] = True

class NeilDecode(NeilBase):
  def __init__(self, set_type, objects, batch_s, scl, n_frames, c, wn_w, wn_h, grav, random_start_pos=False, random_start_speed=False, random_size=False):
    super().__init__(objects, batch_s, scl, n_frames, wn_w, wn_h, grav, random_start_pos, random_start_speed, random_size)

    # Select object static and dynamic properties
    choices    = ['vernier']
    self.ori   = rng().uniform(0, 2*np.pi,      (1, batch_s))
    self.colr  = rng().randint(100, 255,        (c, batch_s))
    self.pop_t = rng().randint(0, n_frames//2,  (1, batch_s))

    self.shape  = rng().choice(choices, (1, batch_s))
    self.side   = rng().randint(0, 2, (1, batch_s)) if len(objects) == 0 else objects[0].side
    self.side_  = 1*self.side                   # evolving value for sqm (deep copy)
    self.popped = np.array([[False]*batch_s])   # display stimulus or not
    self.sizx[self.shape == 'vernier'] /= 1.5   # verniers look better if not too wide
    self.sizy[self.shape == 'vernier'] *= 2.0   # verniers appear smaller than other shapes
    self.pos    = np.vstack((self.x,   self.y))
    self.vel    = np.vstack((self.vx, self.vy))
    self.acc    = np.array([[0.00]*batch_s, [grav]*batch_s])

    # Generate patches to draw the shapes efficiently
    self.patches = []
    for b in range(batch_s):
      patch_info = self.generate_patch_info(self.sizx[0, b], self.sizy[0, b], self.ori[0, b])
      self.patches.append(patch_info)

  def generate_patch_info(self, sizx, sizy, ori):
    max_s   = int(2*max(sizx, sizy))
    patch   = np.zeros((max_s, max_s))
    
    patch_0 = np.zeros((max_s, max_s))  # patch with zero offset

    v_siz_w = rng().uniform(1 + sizx//6, 1 + sizx//2)
    v_siz_h = rng().uniform(1 + sizy//4, 1 + sizy//2)
    v_off_w = rng().uniform(1,              1 + (sizx - v_siz_w)//2)*2
    v_off_h = rng().uniform(1 + v_siz_h//2, 1 + (sizy - v_siz_h)//2)*2

    start1     = (int((max_s - v_off_h - v_siz_h)//2), int((max_s - v_off_w - v_siz_w)//2))
    start2     = (int((max_s + v_off_h - v_siz_h)//2), int((max_s + v_off_w - v_siz_w)//2))
    start01    = (int((max_s - v_off_h - v_siz_h)//2), int((max_s - 0       - v_siz_w)//2))
    start02    = (int((max_s + v_off_h - v_siz_h)//2), int((max_s + 0       - v_siz_w)//2))
    extent     = (int(v_siz_h), int(v_siz_w))
    rr1,  cc1  = rectangle(start=start1,  extent=extent, shape=patch.shape)
    rr2,  cc2  = rectangle(start=start2,  extent=extent, shape=patch.shape)
    rr01, cc01 = rectangle(start=start01, extent=extent, shape=patch.shape)
    rr02, cc02 = rectangle(start=start02, extent=extent, shape=patch.shape)
    patch[  rr1,  cc1 ] = 255
    patch[  rr2,  cc2 ] = 255
    patch_0[rr01, cc01] = 255
    patch_0[rr02, cc02] = 255

    #patch  = rotate(patch, ori).astype(int)
    patch_info = [patch, np.fliplr(patch), patch_0]

    return patch_info

    # Compute what must be updated between the frames
  def compute_changes(self, t, batch_s, objects, cond):
    # Visible objects appear
    self.popped[t >= self.pop_t] = True

    # Decode related changes, offset vernier in same direction with probability p on each frame
    # p1 is probability of PV, p2 is probability of AV
    p1 = 0.4
    p2 = 0
    for b in range(batch_s):
      if t >= self.pop_t[0, b]:
        random_num = random.random()
        if random_num < p1:
          objects[-1].side_[:, b] = self.side[:, b]  # same offset  
        elif random_num < p1 + p2:
          objects[-1].side_[:, b] = 1 - self.side[:, b]      # opposite offset
        else:
          self.side_[:, b] = 2                              # no offset

class NeilSqm(NeilBase):
  def __init__(self, set_type, objects, batch_s, scl, n_frames, c, wn_w, wn_h, grav, pop_t=0, random_start_pos=False, random_start_speed=False, random_size=False):
    super().__init__(objects, batch_s, scl, n_frames, wn_w, wn_h, grav, random_start_pos, random_start_speed, random_size)

    # Select object static and dynamic properties
    choices    = ['vernier'] # type of object that appears in frames
    self.ori   = np.ones((1, batch_s))*0.0 # orientation (rotation)
    self.colr  = np.ones((c, batch_s), dtype=int)*255 # color
    self.pop_t = np.ones((1, batch_s), dtype=int)*pop_t # frame where object appears and seed offset takes place

    self.shape  = rng().choice(choices, (1, batch_s))
    self.side   = rng().randint(0, 2, (1, batch_s)) if len(objects) == 0 else objects[0].side
    self.side_  = 1*self.side                   # evolving value for sqm (deep copy)
    self.popped = np.array([[False]*batch_s])   # display stimulus or not
    self.sizx[self.shape == 'vernier'] /= 1.5   # verniers look better if not too wide
    self.sizy[self.shape == 'vernier'] *= 2.0   # verniers appear smaller than other shapes
    self.pos    = np.vstack((self.x,   self.y))
    self.vel    = np.vstack((self.vx, self.vy))
    self.acc    = np.array([[0.00]*batch_s, [grav]*batch_s])

    # Generate patches to draw the shapes efficiently
    self.patches = []
    for b in range(batch_s):
      patch_info = self.generate_patch_info(self.sizx[0, b], self.sizy[0, b], self.ori[0, b])
      self.patches.append(patch_info)

  def generate_patch_info(self, sizx, sizy, ori):
    max_s   = int(2*max(sizx, sizy))
    patch   = np.zeros((max_s, max_s))

    patch_0 = np.zeros((max_s, max_s))  # patch with zero offset

    v_siz_w =  1 +  sizx//4
    v_siz_h =  1 +  sizy//3
    v_off_w = (1 + (sizx - v_siz_w)//3)*2
    v_off_h = (1 + (sizy - v_siz_h)//6)*2 + v_siz_h//2

    start1     = (int((max_s - v_off_h - v_siz_h)//2), int((max_s - v_off_w - v_siz_w)//2))
    start2     = (int((max_s + v_off_h - v_siz_h)//2), int((max_s + v_off_w - v_siz_w)//2))
    start01    = (int((max_s - v_off_h - v_siz_h)//2), int((max_s - 0       - v_siz_w)//2))
    start02    = (int((max_s + v_off_h - v_siz_h)//2), int((max_s + 0       - v_siz_w)//2))
    extent     = (int(v_siz_h), int(v_siz_w))
    rr1,  cc1  = rectangle(start=start1,  extent=extent, shape=patch.shape)
    rr2,  cc2  = rectangle(start=start2,  extent=extent, shape=patch.shape)
    rr01, cc01 = rectangle(start=start01, extent=extent, shape=patch.shape)
    rr02, cc02 = rectangle(start=start02, extent=extent, shape=patch.shape)
    patch[  rr1,  cc1 ] = 255
    patch[  rr2,  cc2 ] = 255
    patch_0[rr01, cc01] = 255
    patch_0[rr02, cc02] = 255

    patch  = rotate(patch, ori).astype(int)
    patch_info = [patch, np.fliplr(patch), patch_0]

    return patch_info

  # Compute what must be updated between the frames
  def compute_changes(self, t, batch_s, objects, cond):
    # Visible objects appear
    self.popped[t >= self.pop_t] = True

    # Parse the condition to retrieve the first and second offset frame numbers
    pattern_v1 = '^V(\d+)'
    match_v1 = re.search(pattern_v1, cond)
    vernier1_t = int(match_v1.group(1))

    pattern_v2 = '-(PV|AV)(\d+)$'
    match_v2 = re.search(pattern_v2, cond)

    # SQM related changes (TODO make this more elegant)
    for b in range(batch_s):
      if t == vernier1_t:
        objects[-1].side_[:, b] = self.side[:, b]                 # seed offset
      elif match_v2:
        vernier2_t = int(match_v2.group(2))
        vernier2_type = match_v2.group(1)
        if vernier2_t > 0 and t == vernier2_t:
          if vernier2_type == 'AV':
            objects[-1].side_[:, b] = 1 - self.side[:, b]  # opposite offset  
          if vernier2_type == 'PV':
            objects[-1].side_[:, b] = self.side[:, b]      # same offset
        else:
          self.side_[:, b] = 2                               # no offset
      else:
        self.side_[:, b] = 2 # no offset

# Class to generate batches of bouncing balls
class BatchMaker():
  # Initiates all values unchanged from batch to batch
  def __init__(self, set_type, n_objects, batch_s, n_frames, im_dims, condition='V', random_start_pos=False, random_size=False, add_noise=False):
    self.set_type   = set_type
    if set_type == 'recons':
      self.Object = NeilRecons
    elif set_type == 'decode':
      self.Object = NeilDecode
    elif set_type == 'sqm':
      self.Object = NeilSqm
    self.n_objects  = n_objects
    self.n_max_occl = 0
    self.condition  = condition if condition != 'V' else 'V0'  # coding detail
    self.batch_s    = batch_s
    self.n_frames   = n_frames
    self.n_chans    = im_dims[-1]
    self.scale      = max(im_dims[0], im_dims[1])/64
    self.wn_h       = int(im_dims[0]*self.scale)
    self.wn_w       = int(im_dims[1]*self.scale)
    self.gravity    = 0.0
    self.friction   = 0.0

    # Precedence is random_start_pos then random_size
    self.random_start_pos = random_start_pos
    self.random_size = random_size

    self.add_noise = add_noise
  
  # Initialize batch, objects (size, position, velocities, etc.) and background
  def init_batch(self):

    # Background window and objects inside it
    self.batch    = []
    self.objects  = []
    self.window   = 127*np.ones((self.batch_s, self.wn_h, self.wn_w, self.n_chans), dtype=int)
    for _ in range(self.n_objects):
      self.objects.append(self.Object(self.set_type, self.objects, self.batch_s, self.scale,
                  self.n_frames, self.n_chans, self.wn_h, self.wn_w, self.gravity,
                  random_start_pos=self.random_start_pos, random_start_speed=False, random_size=self.random_size))
    self.bg_color = rng().randint(0, 80, (self.batch_s, self.n_chans))  # if set_type == 'recons' else 40*np.ones((self.batch_s, self.n_chans))
    for b in range(self.batch_s):
      for c in range(self.n_chans):
        self.window[b, :, :, c] = self.bg_color[b, c]

    # Occluding walls in the frontground
    n_occl        = rng().randint(0, self.n_max_occl+1, (self.batch_s)) if self.set_type == 'recons' else [0]*self.batch_s
    self.frnt_grd = np.zeros(self.window.shape, dtype=bool)
    for b in range(self.batch_s):
      for _ in range(n_occl[b]):
        if rng().rand() > 0.5:
          pos    = rng().randint(0, self.wn_h)
          height = rng().randint(2, self.wn_h//10)
          self.frnt_grd[b, max(0, pos-height):min(self.wn_h, pos+height), :, :] = True
        else:
          pos    = rng().randint(0, self.wn_w)
          height = rng().randint(2, self.wn_w//10)
          self.frnt_grd[b, :, max(0, pos-height):min(self.wn_w, pos+height), :] = True

  # Batch making function (generating batch_s dynamic sequences)
  def generate_batch(self):
    self.init_batch()
    for t in tqdm(range(self.n_frames)):
  
      # Compute and draw moving objects
      frame = self.window*1
      for i, obj in enumerate(self.objects):
        obj.compute_changes(t, self.batch_s, self.objects, self.condition)
      for obj in self.objects:
        obj.draw(frame, self.batch_s)
      for obj in self.objects:
        obj.update_states(self.batch_s, self.friction)

      # Add noise and black frontground walls
      noise_std = 0.1 * 255.0
      noise = np.random.normal(scale=noise_std, size=frame.shape).clip(0.0, 255.0).astype(np.uint8)
      frame += noise
      frame[self.frnt_grd] = 0.0
      self.batch.append(frame.clip(0, 255).astype(np.uint8))

    # Return batch (and labels)
    if self.set_type == 'recons':
      return self.batch, None                     # list of n_frames numpy arrays of dims [batch, h, w, channels]
    else:
      return self.batch, self.objects[0].side[0]  # the label is always the vernier(s) offset in the first frame


# Show example of reconstruction batch
if __name__ == '__main__':

  import pyglet   # conda install -c conda-forge pyglet
  import imageio  # conda install -c conda-forge imageio
  import os
  
  set_type     = 'sqm'    # 'recons', 'decode' or 'sqm'
  condition    = 'V0-AV1'  # 'V', 'V-PVn' or 'V-AVn', n > 0
  n_objects    = 1 # number of objects in one video sequence
  n_frames     = 13 # length of video sequence in frames
  scale        = 1
  batch_s      = 4 # number of video sequences to generate simultaneously
  n_channels   = 3 # number of channels of video sequences
  batch_maker  = BatchMaker(set_type, n_objects, batch_s, n_frames, (64*scale, 64*scale, n_channels), condition, random_start_pos=True, random_size=True, add_noise=True)
  if set_type == 'recons':
    batch_frames = batch_maker.generate_batch()
  else:
    batch_frames, batch_labels = batch_maker.generate_batch()
  batch_arrays = [np.stack([batch_frames[t][b] for t in range(n_frames)]) for b in range(batch_s)]
  consolidated_array = np.stack([batch_arrays[b] for b in range(batch_s)])
  
  gif_name        = 'test_output.gif'
  display_frames  = []
  for t in range(n_frames):
    display_frames.append(np.hstack([batch_frames[t][b] for b in range(batch_s)]))
  imageio.mimsave(gif_name, display_frames, duration=0.1)
  anim   = pyglet.resource.animation(gif_name)
  sprite = pyglet.sprite.Sprite(anim)
  window = pyglet.window.Window(width=sprite.width, height=sprite.height)
  window.set_location(600, 300)
  @window.event
  def on_draw():
    window.clear()
    sprite.draw()
  pyglet.app.run()
  os.remove(gif_name)