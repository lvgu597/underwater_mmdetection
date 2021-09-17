# model settings
model = dict(
    type='CascadeRCNN',
    # pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        norm_cfg=dict(
            type='BN', #设置bn层
            requires_grad=True),
        norm_eval=True,
        ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8], # 增加scales
            ratios=[0.25, 0.5, 1.0, 2.0, 4],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[1.0, 1.0, 1.0, 1.0])),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
            gc_context=True), #增加gc_context
        bbox_head=[
        dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=4,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=4,
             bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=4,
             bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.033, 0.033, 0.067, 0.067]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    ]))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7, #改成0.6过拟合
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rcnn=[                    # 注意，这里有3个RCNN的模块，对应开头的那个RCNN的stage数量
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    ],
    stage_loss_weights=[1, 0.5, 0.25])     # 3个RCNN的stage的loss权重
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.001, nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.001), max_per_img=100))# 采用soft_nms后处理         # 是否保留所有stage的结果
# dataset settings

dataset_type = 'UnderwaterDataset'
data_root = '/home/jkx/project/smallq/underwater/underwater_mmdetection/coco_data/'  # Root path of data
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'), # First pipeline to load images from file path
    dict(type='LoadAnnotations', with_bbox=True), # Second pipeline to load annotations for current image
    # dict(type='MixUp',p=0.5, lambd=0.5),
    dict(
        type='Resize', # Augmentation pipeline that resize the images and their annotations
        #img_scale=[(4096, 1000)],
        img_scale=[(4096,800),(4096,1200)], #新的scale试下,跑不起来
        #img_scale=[(4096, 800), (4096, 1200)],
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
        # img_scale=[(4096, 800), (4096, 1000), (4096, 1200)], #新的scale试下
        img_scale=(4096, 1000), #最大test scale
        # img_scale=[(600,900),(800,1200),(1000,1500),(1200,1800),(1400,2100)],
        flip=True, #测试时是否翻转图片
        transforms=[
            dict(type='Resize', keep_ratio=True), #保持原始比例的resize
            dict(type='RandomFlip'), #
            dict(type='Normalize', **img_norm_cfg), #规范化
            dict(type='Pad', size_divisor=32), 
            dict(type='ImageToTensor', keys=['img']), #将图片转为tensor
            dict(type='Collect', keys=['img']), #获取关键信息的pipline
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=1,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=1,  #进行oversample操作
        dataset = dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/Duck_inject_no_u_all.json',
            img_prefix=data_root + 'defect_images/',
            pipeline=train_pipeline,
        )),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val0410.json',
        img_prefix=data_root + 'defect_images/',
        pipeline=test_pipeline),
    # test=dict(
    #     type=dataset_type,
    #     ann_file= '/home/jkx/project/smallq/underwater/underwater_mmdetection/tools/data_process/testA.json',
    #     img_prefix=data_root + 'test_images/test-A-image',
    #     pipeline=test_pipeline))
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val0410.json',
        img_prefix=data_root + 'defect_images/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.00375, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[32, 38])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 40
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '../data/work_dirs/cascade_rcnn_r50_fpn_1x_dnu_gc_mixup'
load_from = '../data/work_dirs/cascade_rcnn_r50_fpn_1x_dnu_gc_mixup/latest.pth' #加载预训练网络参数
resume_from = '../data/work_dirs/cascade_rcnn_r50_fpn_1x_dnu_gc_mixup/latest.pth'
workflow = [('train', 1)]