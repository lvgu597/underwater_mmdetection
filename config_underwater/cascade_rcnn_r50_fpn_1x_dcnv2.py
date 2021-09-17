_base_ = './cascade_rcnn_r50_fpn_1x.py'
model = dict(
    backbone = dict(
        norm_cfg=dict(
            type='BN', #设置bn层
            requires_grad=True),
        norm_eval=True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), #dcnv2 c3-c5 
        stage_with_dcn=(False, True, True, True),
    ),
    roi_head = dict(
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
            gc_context=True), #增加gc_context
    ),
)
train_pipeline = [
    dict(type='MixUp',p=0.5, lambd=0.5),
]

work_dir = '../data/work_dirs/cascade_rcnn_r50_fpn_1x_dcnv2'
load_from = '../data/work_dirs/cascade_rcnn_r50_fpn_1x_mixup_gc/latest.pth' #加载预训练网络参数