python train.py ../config/cascade_rcnn_r50_40e_DetectoRS.py --gpus 1
# python train.py ../config/cascade_rcnn_r50_1x_DetectoRS_thr6.py --gpus 1 --no-validate
# python train.py ../config/cascade_rcnn_r50_fpn_1x_gcnet.py --gpus 1 --no-validate
# python train.py ../config/cascade_rcnn_r50_fpn_1x_ghmr.py --gpus 1 --no-validate
# python train.py ../config/cascade_rcnn_r50_fpn_1x_scale.py --gpus 1 --no-validate
# python train.py ../config/cascade_rcnn_x101_1x_DetectoRS.py --gpus 1
# python train.py ../config/cascade_rcnn_r50_fpn_.py --gpus 1
#python ./model_converters/publish_model.py ../data/work_dirs/cascade_rcnn_r50_fpn_1x_duck/latest.pth ../data/work_dirs/cascade_rcnn_r50_fpn_1x_duck/latest-submit.pth
# python publish_model.py ../data/work_dirs/cascade_rcnn_r50_fpn_400/latest.pth ../data/work_dirs/cascade_rcnn_r50_fpn_400/latest-submit.pth