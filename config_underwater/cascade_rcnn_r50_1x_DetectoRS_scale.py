_base_ = 'cascade_rcnn_r50_1x_DetectoRS.py'
# model = dict( #不引入dcnv2试下
#     pretrained='open-mmlab://resnext101_32x4d',
#     backbone=dict(
#         type='DetectoRS_ResNeXt',
#         depth=101,
#         groups=32,
#         base_width=4,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1),
# )
albu_train_transforms = [dict(type='RandomRotate90', always_apply=False, p=0.5)]# 引入随机旋转
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'), # First pipeline to load images from file path
    dict(type='LoadAnnotations', with_bbox=True), # Second pipeline to load annotations for current image
    dict(type='MixUp',p=0.5, lambd=0.5),
    dict(
        type='Resize', # Augmentation pipeline that resize the images and their annotations
        img_scale=[(3840,2160)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5), #翻转 flip_ratio 为翻转概率
    dict(type='Normalize', **img_norm_cfg), #规范化image
    dict(type='Pad', size_divisor=32), # padding设置，填充图片可被32整出除
    dict(type='DefaultFormatBundle'), # Default format bundle to gather data in the pipeline
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']), #决定将哪些关键数据传给detection的管道
]
test_pipeline = [
    dict(type='LoadImageFromFile'), #加载图片的pipline
    #dict(type='Concat', template_path='/tcdata/guangdong1_round2_testB_20191024/'),
    dict(
        type='MultiScaleFlipAug', 
        img_scale=[(3840,2160)], #最大test scale
        # img_scale=[(600,900),(800,1200),(1000,1500),(1200,1800),(1400,2100)],
        flip=True, #测试时是否翻转图片
        transforms=[
            dict(type='Resize', keep_ratio=True), #保持原始比例的resize
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg), #规范化
            dict(type='Pad', size_divisor=32),
            dict(type='Albu', #引入随机旋转
                transforms=albu_train_transforms,
                bbox_params=dict(type='BboxParams',
                                format='pascal_voc',
                                label_fields=['gt_labels'],
                                min_visibility=0.0,
                                filter_lost_elements=True),
                keymap={'img': 'image', 'gt_bboxes': 'bboxes'},
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(type='ImageToTensor', keys=['img']), #将图片转为tensor
            dict(type='Collect', keys=['img']), #获取关键信息的pipline
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        oversample_thr=0.5,  #进行oversample操作
    ),
)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
total_epochs = 12
optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
work_dir = '../data/work_dirs/cascade_rcnn_r50_1x_DetectoRS_scale'
load_from = '../data/work_dirs/cascade_rcnn_r50_1x_DetectoRS/latest.pth'
resume_from = None
