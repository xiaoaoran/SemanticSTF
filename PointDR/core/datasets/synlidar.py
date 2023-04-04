import os
import yaml

import numpy as np

from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from torchpack.utils.logging import logger

__all__ = ['SynLiDAR']


class SynLiDAR(dict):
  def __init__(self, root, voxel_size, num_points, src, sample_stride=1, **kwargs):

    super(SynLiDAR, self).__init__({
      'train':
      SynLiDARInternal(root,
                       voxel_size,
                       num_points,
                       src=src,
                       sample_stride=sample_stride),
      })


class SynLiDARInternal:
  def __init__(self,
               root,
               voxel_size,
               num_points,
               src,
               split='train',
               sample_stride=1,
               pointdr=False):
    self.root = root
    self.split = split
    self.voxel_size = voxel_size
    self.num_points = num_points
    self.src = src
    self.sample_stride = sample_stride
    self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    self.pointdr = pointdr
    print('pointdr: ', pointdr)

    self.files = []
    for seq in self.seqs:
      seq_files = sorted(
        os.listdir(os.path.join(self.root, seq, 'velodyne')))
      seq_paths = [os.path.join(self.root, seq, 'velodyne', x) for x in seq_files]
      self.files.extend(seq_paths)

    # sample scans
    if self.sample_stride > 1:
      self.files.sort()
      files = []
      for ii in range(0, len(self.files), self.sample_stride):
        files.append(self.files[ii])
      self.files = files

    DATA = yaml.safe_load(open('core/datasets/mapping/synlidar.yaml', 'r'))
    remap_dict = DATA["learning_map"]
    max_key = max(remap_dict.keys())
    remap_lut = np.ones((max_key + 100), dtype=np.int32) * 255
    remap_lut[list(remap_dict.keys())] = list(remap_dict.values())
    self.remap_lut = remap_lut
    self.reverse_label_name_mapping = DATA['reverse_label_name_mapping']

    self.num_classes = len(self.reverse_label_name_mapping)
    self.angle = 0.0

  def set_angle(self, angle):
    self.angle = angle

  def __len__(self):
    return len(self.files)

  def __getitem__(self, index):
      return self.return_double_views(index)

  def return_double_views(self, index):
    with open(self.files[index], 'rb') as b:
      block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
    # assign an id for each point for consistency
    ids = np.arange(block_.shape[0])
    # read labels
    label_file = self.files[index].replace('velodyne', 'labels').replace('.bin', '.label')
    if os.path.exists(label_file):
      with open(label_file, 'rb') as a:
        all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
    else:
      all_labels = np.zeros(block_.shape[0]).astype(np.int32)
    labels_ = self.remap_lut[all_labels & 0xFFFF].astype(np.int64)

    #######################
    # original view #
    #######################
    block_1 = block_.copy()
    theta = np.random.uniform(0, 2 * np.pi)
    scale_factor = np.random.uniform(0.95, 1.05)
    rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                        [-np.sin(theta),
                         np.cos(theta), 0], [0, 0, 1]])
    block_1[:, :3] = np.dot(block_1[:, :3], rot_mat) * scale_factor
    if np.random.rand() < 0.5:
        jiterring = np.random.normal(loc=0., scale=0.01, size=(block_.shape[0], 3))
        jiterring = np.clip(jiterring, a_min=-0.05, a_max=0.05)
        block_1[:, :3] += jiterring
    # voxelization
    pc_1_ = np.round(block_1[:, :3] / self.voxel_size).astype(np.int32)
    pc_1_ -= pc_1_.min(0, keepdims=1)

    feat_1_ = block_1
    _, inds_1, inverse_map = sparse_quantize(pc_1_,
                                             return_index=True,
                                             return_inverse=True)
    if len(inds_1) > self.num_points:
      inds_1 = np.random.choice(inds_1, self.num_points, replace=False)

    pc_1 = pc_1_[inds_1]
    feat_1 = feat_1_[inds_1]
    labels_1 = labels_[inds_1]
    ids_1 = ids[inds_1]
    lidar_1 = SparseTensor(feat_1, pc_1)
    labels_1 = SparseTensor(labels_1, pc_1)
    ids_1 = SparseTensor(ids_1, pc_1)
    inverse_map = SparseTensor(inverse_map, pc_1_)

    #######################
    #    augmented view   #
    #######################
    block_2 = block_.copy()
    labels_2 = labels_.copy()
    # aug1: random drop out
    if np.random.random() < 0.5:
      idxes = np.arange(block_.shape[0])
      ratio = np.random.random() * 0.2 + 0.8  # 0.8 - 1
      idxes = np.random.choice(idxes, int(ratio * block_.shape[0]), replace=False)
      block_2 = block_2[idxes]
      labels_2 = labels_2[idxes]
    # aug2: add noise
    if np.random.random() < 0.5:
      xmin, xmax = block_[:, 0].min(), block_[:, 0].max()
      ymin, ymax = block_[:, 1].min(), block_[:, 1].max()
      zmin, zmax = block_[:, 2].min(), block_[:, 2].max()
      imin, imax = block_[:, 3].min(), block_[:, 3].max()
      noise_num = int(np.random.random() * 2000)
      noise_x = np.random.choice(np.arange(xmin, xmax, 0.01), noise_num, replace=True).astype(np.float32)
      noise_y = np.random.choice(np.arange(ymin, ymax, 0.01), noise_num, replace=True).astype(np.float32)
      noise_z = np.random.choice(np.arange(zmin, zmax, 0.01), noise_num, replace=True).astype(np.float32)
      noise_i = np.random.normal(loc=(imin + imax) / 2, scale=0.5, size=noise_num).astype(np.float32)
      noise = np.stack((noise_x, noise_y, noise_z, noise_i), axis=1)
      block_2 = np.concatenate((block_2, noise), axis=0)
      labels_2 = np.concatenate((labels_2, np.ones(noise.shape[0], dtype=np.int64) * 255), axis=0)
      ids = np.concatenate((ids, np.ones(noise.shape[0], dtype=np.int64) * (-1)), axis=0)
    # aug3: rotate and scale
    if np.random.random() < 1.0:
      theta = np.random.uniform(0, 2 * np.pi)
      scale_factor = np.random.uniform(0.95, 1.05)
      rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                          [-np.sin(theta),
                           np.cos(theta), 0], [0, 0, 1]])
      block_2[:, :3] = np.dot(block_2[:, :3], rot_mat) * scale_factor
    # aug4: flip along X axis
    if np.random.rand() < 0.5:
      block_2[:, 0] *= -1
    # aug5: flip along Y axis
    if np.random.rand() < 0.5:
      block_2[:, 1] *= -1
    # aug6: random jittering
    if np.random.rand() < 0.5:
      jiterring = np.random.normal(loc=0., scale=0.01, size=(block_.shape[0], 3))
      jiterring = np.clip(jiterring, a_min=-0.05, a_max=0.05)
      block_[:, :3] += jiterring
    feat_2_ = block_2
    pc_2_ = np.round(block_2[:, :3] / self.voxel_size).astype(np.int32)
    pc_2_ -= pc_2_.min(0, keepdims=1)
    _, inds_2, _ = sparse_quantize(pc_2_,
                                   return_index=True,
                                   return_inverse=True)
    ratio = np.random.random() * 0.2 + 0.8
    if len(inds_2) > int(self.num_points * ratio):
      inds_2 = np.random.choice(inds_2, int(self.num_points * ratio), replace=False)
    pc_2 = pc_2_[inds_2]
    labels_2 = labels_2[inds_2]
    feat_2 = feat_2_[inds_2]
    ids_2 = ids[inds_2]
    lidar_2 = SparseTensor(feat_2, pc_2)
    labels_2 = SparseTensor(labels_2, pc_2)
    ids_2 = SparseTensor(ids_2, pc_2)

    return {
      'lidar': lidar_1,
      'targets': labels_1,
      'inverse_map_dense': inverse_map,
      'file_name': self.files[index],
      'ids_1': ids_1,

      'lidar_2': lidar_2,
      'ids_2': ids_2,
      'targets_2': labels_2
    }

  @staticmethod
  def collate_fn(inputs):
    return sparse_collate_fn(inputs)
