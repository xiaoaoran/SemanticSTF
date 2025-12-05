import os

import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

__all__ = ['SemanticSTF']

learning_map = {
    0: 255,  # "unlabeled",
    1: 0,  # "car",
    2: 1,  # "bicycle",
    3: 2,  # "motorcycle",
    4: 3,  # "truck",
    5: 4,  # "other-vehicle",
    6: 5,  # "person",
    7: 6,  # "bicyclist",
    8: 7,  # "motorcyclist",
    9: 8,  # "road",
    10: 9,  # "parking",
    11: 10,  # "sidewalk",
    12: 11,  # "other-ground",
    13: 12,  # "building",
    14: 13,  # "fence",
    15: 14,  # "vegetation",
    16: 15,  # "trunk",
    17: 16,  # "terrain",
    18: 17,  # "pole",
    19: 18,  # "traffic-sign",
    20: 255  # "invalid"
}

learning_map_inv = { # inverse of previous map
    255: 0,      # "unlabeled", and others ignored
    0: 1,     # "car"
    1: 2,     # "bicycle"
    2: 3,     # "motorcycle"
    3: 4,    # "truck"
    4: 5,     # "other-vehicle"
    5: 6,    # "person"
    6: 7,     # "bicyclist"
    7: 8,     # "motorcyclist"
    8: 9,     # "road"
    9: 10,    # "parking"
    10: 11,    # "sidewalk"
    11: 12,    # "other-ground"
    12: 13,    # "building"
    13: 14,    # "fence"
    14: 15,    # "vegetation"
    15: 16,    # "trunk"
    16: 17,    # "terrain"
    17: 18,    # "pole"
    18: 19    # "traffic-sign"
}


class SemanticSTF(dict):

    def __init__(self, root, voxel_size, num_points, **kwargs):
        submit_to_server = kwargs.get('submit', False)
        sample_stride = kwargs.get('sample_stride', 1)
        google_mode = kwargs.get('google_mode', False)

        if submit_to_server:
            super().__init__({
                'train':
                    SemanticSTFInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=1,
                                          split='train'),
                'test':
                    SemanticSTFInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=1,
                                          split='test')
            })
        else:
            super().__init__({
                'train':
                    SemanticSTFInternal(root,
                                        voxel_size,
                                        num_points,
                                        sample_stride=1,
                                        split='train'),
                'test':
                    SemanticSTFInternal(root,
                                        voxel_size,
                                        num_points,
                                        sample_stride=sample_stride,
                                        split='val')
            })


class SemanticSTFInternal:

    def __init__(self,
                 root,
                 voxel_size,
                 num_points,
                 split,
                 sample_stride=1):
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.sample_stride = sample_stride
        self.seqs = []
        if split == 'train':
            self.seqs = ['train']
        elif self.split == 'val':
            self.seqs = ['val']
        elif self.split == 'test':
            self.seqs = ['test']

        self.files = []
        for seq in self.seqs:
            seq_files = sorted(
                os.listdir(os.path.join(self.root, seq, 'velodyne')))
            seq_files = [
                os.path.join(self.root, seq, 'velodyne', x) for x in seq_files
            ]
            self.files.extend(seq_files)

        if self.sample_stride > 1:
            self.files = self.files[::self.sample_stride]

        remap_dict = learning_map
        max_key = max(remap_dict.keys())
        remap_lut = np.ones((max_key + 100), dtype=np.int32) * 255
        remap_lut[list(remap_dict.keys())] = list(remap_dict.values())
        self.label_map = remap_lut

        self.reverse_label_name_mapping = learning_map_inv
        self.num_classes = 19
        self.angle = 0.0

    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(self.files[index], 'rb') as b:
            block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 5)[:, :4]  # Note: different from SemanticKITTI and SynLiDAR (with four dimensions)
            block_[:, 3] /= 255.
        block = np.zeros_like(block_)

        if 'train' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

            block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor
        else:
            theta = self.angle
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block[...] = block_[...]
            block[:, :3] = np.dot(block[:, :3], transform_mat)

        block[:, 3] = block_[:, 3]
        pc_ = np.round(block[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)

        label_file = self.files[index].replace('velodyne', 'labels').replace('.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            all_labels = np.zeros(pc_.shape[0]).astype(np.int32)

        labels_ = self.label_map[all_labels].astype(np.int64)

        feat_ = block

        _, inds, inverse_map = sparse_quantize(pc_,
                                               return_index=True,
                                               return_inverse=True)

        if 'train' in self.split:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]
        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)

        return {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'file_name': self.files[index]
        }

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
