# 3D Semantic Segmentation in the Wild: Learning Generalized Models for Adverse-Condition Point Clouds

The official implementation of "3D Semantic Segmentation in the Wild: Learning Generalized Models for Adverse-Condition Point Clouds". (CVPR 2023) :fire::fire::fire:

:fire: For more information follow the [PAPER](https://arxiv.org/abs/2304.00690) link!:fire:
 
Authors: [Aoran Xiao](https://scholar.google.com/citations?user=yGKsEpAAAAAJ&hl=zh-EN), [Jiaxing Huang](https://scholar.google.com/citations?user=czirNcwAAAAJ&hl=zh-EN), [Weihao Xuan](https://scholar.google.com/citations?user=7e0W-2AAAAAJ&hl=en&authuser=1&oi=ao), Ruijie Ren, [Kangcheng Liu](https://scholar.google.com/citations?user=qq2aoesAAAAJ&hl=en), [Dayan Guan](https://scholar.google.com/citations?user=9jp9QAsAAAAJ&hl=zh-EN), [Abdulmotaleb El Saddik](https://scholar.google.ca/citations?user=VcOjgngAAAAJ&hl=en), [Shijian Lu](https://personal.ntu.edu.sg/shijian.lu/), [Eric Xing](https://scholar.google.ca/citations?user=5pKTRxEAAAAJ&hl=en&oi=ao)

![image](https://github.com/xiaoaoran/SemanticSTF/blob/master/Img/Picture1.png)

## SemanticSTF dataset
Download SemanticSTF dataset from [GoogleDrive](https://forms.gle/oBAkVJeFKNjpYgDA9), [BaiduYun](https://pan.baidu.com/s/10QqPZuzPclURZ6Niv1ch1g)(code: 6haz). Data folders are as follows:
The data should be organized in the following format:
```
/SemanticSTF/
  └── train/
    └── velodyne
      └── 000000.bin
      ├── 000001.bin
      ...
    └── labels
      └── 000000.label
      ├── 000001.label
      ...
  └── val/
      ...
  └── test/
      ...
  ...
  └── semanticstf.yaml
```
We provide class annotations in 'semanticstf.yaml'

## PointDR
Baseline code for 3D LiDAR Domain Generalization

``
cd pointDR/
``

### Installation

GPU Requirement: > 1 x NVIDIA GeForce RTX 2080 Ti.

The code has been tested with 
 - Python 3.8, CUDA 10.2, Pytorch 1.8.0, TorchSparse 1.4.0. 
 - Python 3.8, CUDA 11.6, Pytorch 1.13.0, TorchSparse 2.0.0b0
 - IMPORTANT: This code base is not compatible with TorchSparse 2.1.0.

Please refer to [here](docs/INSTALL.md) for the installation details.

#### Pip/Venv/Conda
In your virtual environment follow [TorchSparse](https://github.com/mit-han-lab/spvnas). This will install all the base packages.


### Data preparation

#### SynLiDAR
Download SynLiDAR dataset from [here](https://github.com/xiaoaoran/SynLiDAR), then prepare data folders as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    └──sequences/
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        └── 12/
```

#### SemanticKITTI
To download SemanticKITTI follow the instructions [here](http://www.semantic-kitti.org). Then, prepare the paths as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
      └── sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	   ├── 000000.bin
            |   |	   ├── 000001.bin
            |   |	   └── ...
            │   ├── labels/ 
            |   |      ├── 000000.label
            |   |      ├── 000001.label
            |   |      └── ...
            |   ├── calib.txt
            |   ├── poses.txt
            |   └── times.txt
            └── 08/
```

- Don't forget revise the data root dir in  `configs/kitti2stf/default.yaml` and `configs/synlidar2stf/default.yaml`
### Training

For SemanticKITTI->SemanticSTF, run:
```
python train.py configs/kitti2stf/minkunet/cr0p5.yaml
```

For SynLiDAR->SemanticSTF, run:
```
python train.py configs/synlidar2stf/minkunet/cr0p5.yaml
```

### Testing

For SemanticKITTI->SemanticSTF, run:
```
python evaluate.py configs/kitti2stf/minkunet/cr0p5.yaml --checkpoint_path /PATH/CHECKPOINT_NAME.pt
```

For SynLiDAR->SemanticSTF, run:
``` 
python evaluate_by_weather.py configs/synlidar2stf/minkunet/cr0p5.yaml  --checkpoint_path /PATH/CHECKPOINT_NAME.pt
```

You can download the pretrained models on both SemanticKITTI->SemanticSTF and SynLiDAR->SemanticSTF from [here](https://drive.google.com/drive/folders/1GjmAAXMCPrGrCRgYffKNk4cLnG_kbODc?usp=sharing)

## TODO List

- [x] Release of SemanticSTF dataset. :rocket:
- [x] Release of code of PointDR. :rocket:
- [x] Add license. See [here](#license) for more details.
- [ ] Multi-modal UDA for normal-to-adverse weather 3DSS.

## References

If you find our work useful in your research, please consider citing:  
```
@article{xiao20233d,
  title={3D Semantic Segmentation in the Wild: Learning Generalized Models for Adverse-Condition Point Clouds},
  author={Xiao, Aoran and Huang, Jiaxing and Xuan, Weihao and Ren, Ruijie and Liu, Kangcheng and Guan, Dayan and Saddik, Abdulmotaleb El and Lu, Shijian and Xing, Eric},
  journal={arXiv preprint arXiv:2304.00690},
  year={2023}
}
```
SemanticSTF dataset consists of re-annotated LiDAR point cloud data from the STF dataset. Kindly consider citing it if you intend to use the data:
```
@inproceedings{bijelic2020seeing,
  title={Seeing through fog without seeing fog: Deep multimodal sensor fusion in unseen adverse weather},
  author={Bijelic, Mario and Gruber, Tobias and Mannan, Fahim and Kraus, Florian and Ritter, Werner and Dietmayer, Klaus and Heide, Felix},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11682--11692},
  year={2020}
}
```

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Recommended Repos
Check our other repos for point cloud understanding!
- [Learning From Synthetic LiDAR Sequential Point Cloud for Semantic Segmentation (AAAI2022)](https://github.com/xiaoaoran/SynLiDAR)
- [PolarMix: A General Data Augmentation Technique for LiDAR Point Clouds (NeurIPS 2022)](https://github.com/xiaoaoran/polarmix)
- [Unsupervised Point Cloud Representation Learning with Deep Neural Networks: A Survey (TPAMI2023)](https://github.com/xiaoaoran/3d_url_survey)

## Thanks
We thank the opensource projects [TorchSparse](https://github.com/mit-han-lab/torchsparse), [SPVNAS](https://github.com/mit-han-lab/spvnas) and [SeeingThroughFog](https://github.com/princeton-computational-imaging/SeeingThroughFog).