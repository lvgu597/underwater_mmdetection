_base_ = './cascade_rcnn_r50_fpn_1x.py'
model = dict(
    backbone = dict(
        norm_cfg=dict(
            type='BN', #设置bn层
            requires_grad=True),
        norm_eval=True,
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False), #dcn v1
        stage_with_dcn=(False, True, True, True),
    )
)

work_dir = '../data/work_dirs/cascade_rcnn_r50_fpn_1x_dcnv1'
load_from = '../data' #加载预训练网络参数