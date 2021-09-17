_base_ = './cascade_rcnn_r50_fpn_1x_duck.py'
train_pipeline = [
    dict(type='MixUp',p=0.5, lambd=0.5),
]
work_dir = '../data/work_dirs/cascade_rcnn_r50_fpn_1x_duck_mixup'
load_from = '../data/work_dirs/cascade_rcnn_r50_fpn_1x_duck/latest.pth' #加载预训练网络参数