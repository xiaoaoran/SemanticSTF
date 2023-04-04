import os

import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from torchpack.utils.logging import logger

__all__ = ['SemanticKITTI']

label_name_mapping = {
    0: 'unlabeled',
    1: 'outlier',
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    16: 'on-rails',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    40: 'road',
    44: 'parking',
    48: 'sidewalk',
    49: 'other-ground',
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',
    71: 'trunk',
    72: 'terrain',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    256: 'moving-on-rails',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}

kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]


class SemanticKITTI(dict):

    def __init__(self, root, voxel_size, num_points, **kwargs):
        submit_to_server = kwargs.get('submit', False)
        sample_stride = kwargs.get('sample_stride', 1)
        google_mode = kwargs.get('google_mode', False)

        logger.info("SKT")

        if submit_to_server:
            super().__init__({
                'train':
                    SemanticKITTIInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=1,
                                          split='train',
                                          submit=True),
                'test':
                    SemanticKITTIInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=1,
                                          split='test')
            })
        else:
            super().__init__({
                'train':
                    SemanticKITTIInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=1,
                                          split='train',
                                          google_mode=google_mode),
                'test':
                    SemanticKITTIInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=sample_stride,
                                          split='val')
            })


class SemanticKITTIInternal:

    def __init__(self,
                 root,
                 voxel_size,
                 num_points,
                 split,
                 sample_stride=1,
                 submit=False,
                 google_mode=True):
        if submit:
            trainval = True
        else:
            trainval = False
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.sample_stride = sample_stride
        self.google_mode = google_mode
        self.seqs = []
        if split == 'train':
            self.seqs = [
                '00', '01', '02', '03', '04', '05', '06', '07', '09', '10'
            ]
            if self.google_mode or trainval:
                self.seqs.append('08')
        elif self.split == 'val':
            self.seqs = ['08']
        elif self.split == 'test':
            self.seqs = [
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
            ]

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

        reverse_label_name_mapping = {}
        self.label_map = np.zeros(260)
        cnt = 0
        for label_id in label_name_mapping:
            if label_id > 250:
                if label_name_mapping[label_id].replace('moving-',
                                                        '') in kept_labels:
                    self.label_map[label_id] = reverse_label_name_mapping[
                        label_name_mapping[label_id].replace('moving-', '')]
                else:
                    self.label_map[label_id] = 255
            elif label_id == 0:
                self.label_map[label_id] = 255
            else:
                if label_name_mapping[label_id] in kept_labels:
                    self.label_map[label_id] = cnt
                    reverse_label_name_mapping[
                        label_name_mapping[label_id]] = cnt
                    cnt += 1
                else:
                    self.label_map[label_id] = 255

        self.reverse_label_name_mapping = reverse_label_name_mapping
        self.num_classes = cnt
        self.angle = 0.0

    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.files)

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
        labels_ = self.label_map[all_labels & 0xFFFF].astype(np.int64)

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
        # voxelization
        pc_1_ = np.round(block_1[:, :3] / self.voxel_size).astype(np.int32)
        pc_1_ -= pc_1_.min(0, keepdims=1)

        feat_1_ = block_1
        _, inds_1, inverse_map = sparse_quantize(pc_1_,
                                                 return_index=True,
                                                 return_inverse=True)
        if len(inds_1) > self.num_points:
            inds_1 = np.random.choice(inds_1, self.num_points, replace=False)  # Note this step causes cuda problem if evaluating

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

    def __getitem__(self, index):
       # return double views for contrastive learning
       return self.return_double_views(index)


    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
