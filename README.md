# RECVIS-transporter-networks

```{bash}
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda-10-2
```

```{bash}
cd /usr/local
sudo rm cuda
ln -s cuda-10.1 cuda
```

```{bash}
# install the project requirements
pip install -r requirements.txt

# install ravens
pip install -r ravens/requirements.txt
pip install -e ravens/ # editable version

# install the toolbox for depth estimation
pip install -e Monocular-Depth-Estimation-Toolbox # editable version
# get pretrained model
mkdir checkpoints && cd checkpoints
gdown https://drive.google.com/uc\?id\=1tcWx_BQBNJHpP5-RUWGWjpVRfeUiUMzJ
mim install mmcv-full==1.4.0
```


