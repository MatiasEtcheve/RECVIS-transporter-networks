# RECVIS-transporter-networks

```{bash}
conda create -n RECVIS python==3.7.12 -c conda-forge -y
conda activate RECVIS
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y
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
mkdir checkpoints
gdown https://drive.google.com/uc\?id\=1tcWx_BQBNJHpP5-RUWGWjpVRfeUiUMzJ -0 checkpoints
mim install mmcv-full==1.4.0
```


