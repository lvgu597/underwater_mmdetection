from mmdet.apis import init_detector, inference_detector
import mmcv

config_file = '../config/cascade_rcnn_r50_40e_DetectoRS.py'
checkpoint_file = '../data/work_dirs/cascade_rcnn_r50_40e_DetectoRS/latest.pth'

#初始化模型
model = init_detector(config_file, checkpoint_file)

#测试一张图片
img = '0.tif'
result = inference_detector(model, img)
model.show_result(img, result, out_file='result.tif')

