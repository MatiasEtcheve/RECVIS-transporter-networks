_base_ = [
    "../Monocular-Depth-Estimation-Toolbox/configs/_base_/models/adabins.py",
    # "Monocular-Depth-Estimation-Toolbox/configs/_base_/datasets/nyu.py",
    "../Monocular-Depth-Estimation-Toolbox/configs/_base_/default_runtime.py",
    "../Monocular-Depth-Estimation-Toolbox/configs/_base_/schedules/schedule_24x.py",
]

norm_cfg = dict(type="BN", requires_grad=True)

model = dict(
    type="DepthEncoderDecoder",
    backbone=dict(type="EfficientNet"),
    decode_head=dict(
        type="AdabinsHead",
        in_channels=[24, 40, 64, 176, 2048],
        up_sample_channels=[128, 256, 512, 1024, 2048],
        channels=128,
        align_corners=True,
        loss_decode=dict(type="SigLoss", valid_mask=True, loss_weight=10),
        min_depth=0.001,
        max_depth=10,
        norm_cfg=norm_cfg,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
find_unused_parameters = True
SyncBN = True

# dataset settings Only for test
dataset_type = "CustomDepthDataset"
data_root = "dataset/depth/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (416, 544)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="DepthLoadAnnotations"),
    dict(type="NYUCrop", depth=True),
    dict(type="RandomRotate", prob=0.5, degree=2.5),
    dict(type="RandomFlip", prob=0.5),
    dict(type="RandomCrop", crop_size=(416, 544)),
    dict(
        type="ColorAug",
        prob=1,
        gamma_range=[0.9, 1.1],
        brightness_range=[0.75, 1.25],
        color_range=[0.9, 1.1],
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "depth_gt"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", img_scale=(480, 640)),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(480, 640),
        flip=True,
        flip_direction="horizontal",
        transforms=[
            dict(type="RandomFlip", direction="horizontal"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root + "/train",
        depth_scale=1,
        pipeline=train_pipeline,
        min_depth=1e-3,
        max_depth=10,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root + "/val",
        depth_scale=1,
        pipeline=test_pipeline,
        min_depth=1e-3,
        max_depth=10,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root + "/val",
        pipeline=test_pipeline,
        min_depth=1e-3,
        max_depth=10,
    ),
)
dist_params = dict(backend="nccl")
log_level = "INFO"
workflow = [("train", 1)]

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
# optimizer
max_lr = 0.000357
optimizer = dict(
    type="AdamW",
    lr=max_lr,
    betas=(0.95, 0.99),
    weight_decay=0.1,
    paramwise_cfg=dict(
        custom_keys={
            "decode_head": dict(lr_mult=10),  # x10 lr
        }
    ),
)

# learning policy
lr_config = dict(
    policy="OneCycle",
    max_lr=max_lr,
    div_factor=25,
    final_div_factor=100,
    by_epoch=False,
)
momentum_config = dict(policy="OneCycle")

# runtime
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
evaluation = dict(by_epoch=True, interval=1)

# runtime settings
runner = dict(type="EpochBasedRunner", max_iters=24)
checkpoint_config = dict(by_epoch=True, max_keep_ckpts=2, interval=1)

# iter runtime
log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),  # TensorboardImageLoggerHook
    ],
)
load_from = "adabins_efnetb5_nyu.pth"
resume_from = None
