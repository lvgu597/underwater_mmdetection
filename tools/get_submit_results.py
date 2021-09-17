import time, os
import json
import mmcv 
from mmdet.apis import init_detector, inference_detector

def main():

    config_file = '../config/cascade_rcnn_r50_40e_DetectoRS.py'  # 修改成自己的配置文件
    checkpoint_file = '../data/work_dirs/cascade_rcnn_r50_40e_DetectoRS/latest.pth' # 修改成自己的训练权重

    test_path = '/home/jkx/project/smallq/underwater/underwater_mmdetection/coco_data/test_images/test-A-image'  # 官方测试集图片路径

    csv_name = "result_"+""+time.strftime("%Y%m%d%H%M%S", time.localtime())+".csv"
    
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    classes = {1: 'Port', 2: 'Airport'}
    img_list = []
    for img_name in os.listdir(test_path):
        if img_name.endswith('.jpg'):
            img_list.append(img_name)

    csv_file = open(csv_name, 'w')
    # csv_file.write("name,image_id,confidence,xmin,ymin,xmax,ymax\n")
    for i, img_name in enumerate(img_list, 1):
        # print(i, img_name)
        full_img = os.path.join(test_path, img_name)
        predict = inference_detector(model, full_img)
        for i, bboxes in enumerate(predict, 1):
            # print(i, bboxes)
            if len(bboxes)>0:
                defect_label = classes.get(i)
                image_name = img_name.replace('.jpg','')
                print(defect_label, image_name)
                for bbox in bboxes:
                    xmin, ymin, w, h, confidence = bbox.tolist()
                    xmax = xmin + w
                    ymax = ymin + h
                    csv_file.write(defect_label + ',' + image_name + ',' + str(confidence) + ',' + str(int(xmin)) + ',' + str(int(ymin)) + ',' + str(int(xmax)) + ',' + str(int(ymax)) + '\n')
    csv_file.close()
        
if __name__ == "__main__":
    main()