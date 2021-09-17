_base_ = './cascade_rcnn_r50_fpn_1x.py'
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.6,
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
)

work_dir = '../data/work_dirs/cascade_rcnn_r50_fpn_1x_thr6'
load_from = '../data/work_dirs/cascade_rcnn_r50_fpn_1x/latest.pth' #加载效果更好的 原始/增广模型
