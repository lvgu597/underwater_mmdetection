_base_ = './cascade_rcnn_r50_fpn_1x.py'
train_cfg = dict(
    rcnn=[                    # 注意，这里有3个RCNN的模块，对应开头的那个RCNN的stage数量
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.4,
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
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
            debug=False)
    ],
)
albu_train_transforms = [dict(type='RandomRotate90', always_apply=False, p=0.5)]# 随机旋转90°取得最优成绩
work_dir = '../data/work_dirs/cascade_rcnn_r50_fpn_1x_456'
load_from = '../data/work_dirs/cascade_rcnn_r50_fpn_1x/latest.pth' #加载效果更好的 原始/增广模型
