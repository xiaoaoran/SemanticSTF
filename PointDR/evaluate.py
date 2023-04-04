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
from torchpack.utils.logging import logger
from tqdm import tqdm

from core import builder
from core.callbacks import MeanIoU


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
            print(ious)
            print(miou)
        return miou, ious


def main() -> None:
    # dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--checkpoint_path', help='checkpoint_path')
    parser.add_argument('--name', type=str, default='minkunet', help='model name')
    parser.add_argument('--save_pred', type=str, default=None, help='save prediction dir, do not save if none')
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

    mIoU = MeanIoU_cus(name=f'iou/test_' + checkpoint_path, num_classes=configs.data.num_classes, ignore_label=configs.data.ignore_label)
    mIoU.before_epoch()

    model.eval()

    if args.save_pred is not None:
        os.makedirs(args.save_pred)

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
        mIoU.after_step(output_dict)

        if args.save_pred is not None:
            outputs_np = outputs.cpu().numpy().astype(np.int32)
            filename = feed_dict['file_name']
            filename = os.path.basename(filename[0]).replace('.bin', '.label')
            outputs_np.tofile(args.save_pred+'/'+filename)

    # trainer.after_epoch()
    miou, ious = mIoU.after_epoch()

    print("===" * 10)
    print(dataset)
    print("iou per class: ", ious)
    print("miou:", miou)
    print("===" * 10)

if __name__ == '__main__':
    main()
