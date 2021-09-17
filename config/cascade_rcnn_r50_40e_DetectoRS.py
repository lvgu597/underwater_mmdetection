_base_ = './cascade_rcnn_r50_fpn_1x.py'
model = dict(
    backbone=dict(
        type='DetectoRS_ResNet',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True,
        # dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), #dcnv2 c3-c5
        ),
    roi_head = dict(
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],)
            # gc_context=True), #增加gc_context
    ),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            pretrained='torchvision://resnet50',
            style='pytorch')))
# train_pipeline = [
#     dict(type='MixUp',p=0.5, lambd=0.5),
# ]
data = dict(
    samples_per_gpu=1,
)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[32, 38])  # 20 16,19 12 8,11
total_epochs = 40
optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
work_dir = '../data/work_dirs/cascade_rcnn_r50_40e_DetectoRS'
load_from = '../data/pretrained/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth'
resume_from = None
