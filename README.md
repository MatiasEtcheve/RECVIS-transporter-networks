# RECVIS-transporter-networks

This repository is my work on the course *Object Recognition and Computer Vision* given at the MVA by Jean Ponce, Ivan Laptev, Cordelia Schmid and Josef Sivic.


## Objective

### Transporter Networks

Firstly, we use the Transporter Networks published in this [article](https://arxiv.org/abs/2010.14406). 
This new type of network aims at achieving state-of-the-art performances on robotic manipulation tasks. 
This idea is to decompose a robotic manipulation in 2 steps:
* a pick step where the robot picks an object. Transporter Networks uses an equivariant attention network based on Resnet for such steps.
* a place step conditionned by the pick step where the robot puts down the picked object. 
Transporter networks uses an action-value function invariant to the pick step.
Thanks to the 2 equivariant and invariant properties, the transporter networks are highly sample-efficient.

The authors have published [ravens](https://github.com/google-research/ravens/tree/master/ravens): a python framework to simulate
10 robot manipulation tasks. In this work, we will focus on 2 tasks:
* `block-insertion`: the robot needs to pick a L-shaped object and put it on a L-shaped support
* `manipulating-rope`: the robot needs to manipulate a rope so that it finishes the incomplete
perimeter of a square.

### Depth estimation of the transporter network inputs

Transport networks uses RGB-Depth images. While obtaining depth images is
getting easier, it is still preferable to only use RGB images.
We use the BinsFormer framework published in this [article](https://arxiv.org/abs/2204.00987).
This algorithm predicts depth on RGB images.

The authors have published the algorithm inside this [toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox).

## Installation

The repository uses 2 different packages:
* `ravens`: a gym-like framework which implements transporter networks. 
It uses `tensorflow==2.3`, which works best wih `cuda-10.1`. 
This version of cuda is going to be the primary one.
* `Monocular-Depth-Estimation-Toolbox`: a toolbox which implements Adabins and BinsFormer. 
It uses `torch==1.8`, which works best/only with `cuda-10.2` 
This version of cuda is going to be the secondary one. 

In the installation process, you will need to:
* install the nvidia drivers
* install `cudnn7`, used for `cuda-10.1`
* install 2 versions of cuda. Check this [link](https://medium.com/@peterjussi/multicuda-multiple-versions-of-cuda-on-one-machine-4b6ccda6faae)
  * install `cuda-10.1`
  * install `cuda-10.2`
  * update your env variable `$LD_LIBRARY_PATH`
  * update your symlink at `/usr/local/cuda` so that it points towards `/usr/local/cuda-10.1`

### Installing cuda

Basic script to install `cuda-10.2`. Check this [link](https://developer.nvidia.com/cuda-toolkit-archive) for more information.
You also need to install `cuda-10.1`.

```{bash}
# Install cuda-10.2
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda-10-2
```

```{bash}
# Add symlink to cuda-10.1
cd /usr/local
sudo rm cuda
ln -s cuda-10.1 cuda
```

### Install Pytorch3d

```{bash}
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Installation of the packages

Then you can simply install the repository and its submodules

```{bash}
# Clone the repository 
git clone --recurse-submodules git@github.com:MatiasEtcheve/RECVIS-transporter-networks.git

# install the project requirements
pip install -r requirements.txt

# install ravens
pip install -r ravens/requirements.txt
pip install -e ravens/ # editable version

# install the toolbox for depth estimation
pip install -e Monocular-Depth-Estimation-Toolbox/ # editable version
mim install mmcv-full==1.4.0
```

> **Troubleshooting**:
`mimcv-full` may have some troubles to get installed. Uninstalling and reinstalling `openmim` may work.

### Download the pretrained model checkpoint

The repository uses BinsFormer pretrained model. You need to save it (in `checkpoints/` for instance.)

```{bash}
mkdir checkpoints && cd checkpoints
gdown https://drive.google.com/uc\?id\=1tcWx_BQBNJHpP5-RUWGWjpVRfeUiUMzJ
```


