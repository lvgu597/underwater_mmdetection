_base_ = './cascade_rcnn_r50_fpn_1x.py'
model = dict(
    backbone=dict(plugins=[
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 4),
            stages=(False, True, True, True),
            position='after_conv3')
    ]))
work_dir = '../data/work_dirs/cascade_rcnn_r50_fpn_1x_gcnet'
load_from = '../data/work_dirs/cascade_rcnn_r50_fpn_1x/latest.pth' #加载效果更好的 原始/增广模型
