# Installation Guide

### Prerequisites
Ensure you have the following specifications:
- PyTorch 1.13.0
- CUDA 11.6
- Python 3.8
- TorchSparse 2.0.0b or 1.4.0 (< 2.1.0)

### Installation Steps

1. **Setting up a Conda Environment**:  
   We recommend establishing a new conda environment for this installation.
```
$ conda create -n pointdr python=3.8
$ conda activate pointdr
```
2. **Installing PyTorch**:  
Install PyTorch, TorchVision with specific CUDA support.
```
$ pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```
3. **Additional Dependencies**:  
Install additional utilities and dependencies.
```
$ pip install tqdm

$ sudo apt-get update
$ sudo apt-get install libsparsehash-dev

$ conda install backports
```
4. **Installing TorchSparse**:  
Update and install TorchSparse from its GitHub repository.
```
$ pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git
```