import os
import argparse
import numpy as np

import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm

from core import builder
from core.callbacks import MeanIoU

def count_weather(data_root):
    # get all weather info
    dense_fog_list = []
    light_fog_list = []
    snow_list = []
    rain_list = []
    file = open(data_root+'/val/val.txt', 'r')
    file_names = file.readlines()
    file.close()
    for path in file_names:
        name = path.split(',')[0]
        weather = path.split(',')[1].split('\n')[0]
        if weather == 'dense_fog':
              dense_fog_list.append(name)
        elif weather == 'light_fog':
              light_fog_list.append(name)
        elif weather == 'snow':
              snow_list.append(name)
        elif weather == 'rain':
              rain_list.append(name)
        else:
            raise ValueError

    return dense_fog_list, light_fog_list, snow_list, rain_list


class MeanIoU_cus(MeanIoU):
    def __init__(self,
                 num_classes: int,
                 ignore_label: int,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'iou') -> None:
        super().__init__(num_classes=num_classes,
                         ignore_label=ignore_label,
                         output_tensor=output_tensor,
                         target_tensor=target_tensor,
                         name=name)

    def after_epoch(self):
        return self._after_epoch()

    def _after_epoch(self):
        for i in range(self.num_classes):
            self.total_seen[i] = dist.allreduce(self.total_seen[i],
                                                reduction='sum')
            self.total_correct[i] = dist.allreduce(self.total_correct[i],
                                                   reduction='sum')
            self.total_positive[i] = dist.allreduce(self.total_positive[i],
                                                    reduction='sum')

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                ious.append(cur_iou)

        miou = np.mean(ious)
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar(self.name, miou * 100)
        else:
            ious_ = ['{:.1f}'.format(x*100) for x in ious]
            ious_ = ' & '.join(ious_)
            print(ious_)
            print(miou*100)
        return miou, ious


def main() -> None:
    # dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--checkpoint_path', help='checkpoint_path')
    parser.add_argument('--name', type=str, default='minkunet', help='model name')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    dataset = builder.make_dataset()
    dataflow = {}
    for split in dataset:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.batch_size if split == 'train' else 1,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    assert configs.model.name == 'minkunet'

    model = builder.make_model().cuda()

    checkpoint_path = args.checkpoint_path

    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['model'])
    model = model.cuda()
    print("\nModel loaded from {}\n".format(checkpoint_path))

    dense_fog_list, light_fog_list, snow_list, rain_list = count_weather(data_root=configs.tgt_dataset.root)

    mIoU_dense_fog = MeanIoU_cus(name=f'iou/test_dense_fog_' + checkpoint_path, num_classes=configs.data.num_classes, ignore_label=configs.data.ignore_label)
    mIoU_light_fog = MeanIoU_cus(name=f'iou/test_light_fog_' + checkpoint_path, num_classes=configs.data.num_classes, ignore_label=configs.data.ignore_label)
    mIoU_rain = MeanIoU_cus(name=f'iou/test_rain_' + checkpoint_path, num_classes=configs.data.num_classes, ignore_label=configs.data.ignore_label)
    mIoU_snow = MeanIoU_cus(name=f'iou/test_snow_' + checkpoint_path, num_classes=configs.data.num_classes, ignore_label=configs.data.ignore_label)
    mIoU_dense_fog.before_epoch()
    mIoU_light_fog.before_epoch()
    mIoU_rain.before_epoch()
    mIoU_snow.before_epoch()

    model.eval()

    for feed_dict in tqdm(dataflow['test'], desc='eval'):
        _inputs = {}
        for key, value in feed_dict.items():
            if 'name' not in key:
                _inputs[key] = value.cuda()

        inputs = _inputs['lidar']
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
        outputs, _ = model(inputs)

        invs = feed_dict['inverse_map']
        all_labels = feed_dict['targets_mapped']
        _outputs = []
        _targets = []
        for idx in range(invs.C[:, -1].max() + 1):
            cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
            cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
            cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
            outputs_mapped = outputs[cur_scene_pts][cur_inv].argmax(1)
            targets_mapped = all_labels.F[cur_label]
            _outputs.append(outputs_mapped)
            _targets.append(targets_mapped)
        outputs = torch.cat(_outputs, 0)
        targets = torch.cat(_targets, 0)
        output_dict = {'outputs': outputs, 'targets': targets}
        # trainer.after_step(output_dict)
        file_name = feed_dict['file_name']
        file_name = os.path.basename(file_name[0]).split('.')[0]
        if file_name in dense_fog_list:
            mIoU_dense_fog.after_step(output_dict)
        elif file_name in light_fog_list:
            mIoU_light_fog.after_step(output_dict)
        elif file_name in rain_list:
            mIoU_rain.after_step(output_dict)
        elif file_name in snow_list:
            mIoU_snow.after_step(output_dict)
        else:
            raise ValueError

    # trainer.after_epoch()
    print("===" * 10)
    print('Dense_fog')
    miou_dense_fog, ious = mIoU_dense_fog.after_epoch()
    print(ious)
    ious = np.asarray(ious)
    mask = (ious != 1)
    ious = ious[mask]
    print("Valid class num: ", mask.sum())
    miou_dense_fog = ious.mean()
    print("===" * 10)
    print('Light_fog')
    miou_light_fog, ious = mIoU_light_fog.after_epoch()
    print(ious)
    ious = np.asarray(ious)
    mask = (ious != 1)
    ious = ious[mask]
    miou_light_fog = ious.mean()
    print("Valid class num: ", mask.sum())
    print("===" * 10)
    print('Rain')
    miou_rain, ious = mIoU_rain.after_epoch()
    print(ious)
    ious = np.asarray(ious)
    mask = (ious != 1)
    ious = ious[mask]
    miou_rain = ious.mean()
    print("Valid class num: ", mask.sum())
    print("===" * 10)
    print('Snow')
    miou_snow, ious = mIoU_snow.after_epoch()
    print(ious)
    ious = np.asarray(ious)
    mask = (ious != 1)
    ious = ious[mask]
    miou_snow = ious.mean()
    print("Valid class num: ", mask.sum())

    print("===" * 10)
    print('dense_fog, light_fog, rain, snow: {:.1f} {:.1f} {:.1f} {:.1f}'
          .format(miou_dense_fog*100, miou_light_fog*100, miou_rain*100, miou_snow*100))

if __name__ == '__main__':
    main()
