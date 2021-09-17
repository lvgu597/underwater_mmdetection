_base_ = './cascade_rcnn_r50_fpn_1x.py'
model = dict(
    rpn_head = dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4,8],
            ratios=[0.25, 0.5, 1.0, 2.0, 4],
            strides=[4, 8, 16, 32, 64]),
    )
)
work_dir = '../data/work_dirs/cascade_rcnn_r50_fpn_1x_scale'
load_from = '../data/work_dirs/cascade_rcnn_r50_fpn_1x/latest.pth' #加载效果更好的 原始/增广模型
