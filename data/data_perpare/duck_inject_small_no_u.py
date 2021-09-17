'''
将类别比较少的holothurian进行填鸭式处理
具体操作抠到no_target_images中
'''
# 填鸭式操作,对数据集比较少、识别率不高的进行增强
import os
import json
import numpy as np
import pandas as pd
import cv2
import glob
import random
from PIL import Image
import time
from sklearn import metrics as mr
import logging
random.seed(200)
classes = {'echinus': 1, 'starfish': 2, 'holothurian': 3, 'scallop': 4}
defect_img_root='../../coco_data/defect_images/'
normal_img_paths = os.listdir('../../coco_data/no_target_images/')
save_dir='../../coco_data/defect_images/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
# normal_img_root='./normal_images/'
# aug_dir='./normal_images_aug/'
anno_file='../../coco_data/annotations/instances_trainall.json'
with open(anno_file, 'r') as f:
    anno_result = json.load(f)
name_list = []
for file_name in anno_result.get('images'):  #file_name img_id 为其index
    name_list.append(file_name.get('file_name'))
#print(anno_result.get('annotations')) 
anno_pd = pd.DataFrame(anno_result.get('annotations'))
ring_width=10# default is 5
img_id = anno_result.get('images')[-1].get('id')
anno_id = anno_result.get('annotations')[-1].get('id')
last_result_length=0
img_name_count=0
for index in range(len(name_list)):
    save_temp_flag = False
    img_path = f'{defect_img_root}{name_list[index]}'
    if 'u' not in img_path: # 不扩充带u的数据
        img_anno = anno_pd[anno_pd['image_id'] == index] #获取该image_id 对应anno信息
        bboxs =  img_anno["bbox"].tolist()#应该根据img_id查找
        defect_names = img_anno["category_id"].tolist()
        # print(img_path)
        testimg = Image.open(img_path)
        normal_img_path = random.sample(normal_img_paths, 1)[0] #随机从no_target_images中取一张作为模板照片
        # print(normal_img_path)
        temp_img = Image.open(f'../../coco_data/no_target_images/{normal_img_path}') 
        save_temp_name = 'template_small_no_u_all_' + name_list[index].strip('.jpg') + '_' +str(index) + '.jpg'
        # print(index)
        logging.info(index)
        for idx in range(len(bboxs)):
            left_top_x_list = [testimg.size[0]/len(bboxs)*i for i in range(len(bboxs)+1)] #将原始x划分成len(bbox)个区域
            left_top_y_list = [testimg.size[1]/len(bboxs)*i for i in range(len(bboxs)+1)]
            # print(left_top_x_list, left_top_y_list)
            pts=bboxs[idx]
            d_name=defect_names[idx] #类别,即category id
            xmin=pts[0]
            ymin=pts[1]
            defect_w=pts[2]
            defect_h=pts[3]
            xmax=abs(xmin+defect_w)
            ymax=abs(ymin+defect_h)
            w_h=round(defect_w/defect_h,2)
            h_w=round(defect_h/defect_w,2)
            # 筛选出小目标
            if defect_w <= 384 and defect_h <= 206:
                left_top_x=int(random.uniform(left_top_x_list[idx],left_top_x_list[idx+1]))
                left_top_y=int(random.uniform(left_top_y_list[idx],left_top_y_list[idx+1]))
                #print(left_top_x, left_top_y)
                logging.info(left_top_x, left_top_y)
                #print(left_top_x,left_top_y)
                mask=np.zeros_like(temp_img)
                mask[int(left_top_y-ring_width):int(left_top_y+defect_w+ring_width),int(left_top_x-ring_width):int(left_top_x+defect_h+ring_width)]=255
                mask[int(left_top_y):int(left_top_y+defect_w),int(left_top_x):int(left_top_x+defect_h)]=0

                # cv2.namedWindow("mask",0);
                # cv2.resizeWindow("mask", 1200, 800);
                # cv2.imshow('mask',mask)
                # cv2.imwrite('mask.jpg',mask)
                # cv2.waitKey(0)
                patch=testimg.crop((xmin,ymin,xmax,ymax))
                #====相似度计算==============================================================================================#
                patch1=patch.copy()
                patch2=temp_img.crop((left_top_x,left_top_y,int(left_top_x+patch1.size[0]),int(left_top_y+patch1.size[1])))

                # print('bbox:',(left_top_x,left_top_y,int(left_top_x+(xmax-xmin)),int(left_top_y+(ymax-ymin))))
                # print(patch1.size[0],patch1.size[1])
                # print(patch1.size,patch2.size)
                patch2.resize((patch1.size[0],patch1.size[1]))
                patch1=np.resize(patch1,-1)
                patch2=np.resize(patch2,-1)
                # print(patch1.shape)
                # print(patch2.shape)
                mutual_infor=mr.mutual_info_score(patch1,patch2)
                # print(mutual_infor)
                #==================================================================================================#
                if mutual_infor>0.8:
                    temp_img.paste(patch,(left_top_x,left_top_y))
                    temp_img = cv2.cvtColor(np.asarray(temp_img),cv2.COLOR_RGB2BGR)
                    temp_img = cv2.inpaint(temp_img,mask[:,:,0],3,cv2.INPAINT_TELEA)
                    temp_img = Image.fromarray(cv2.cvtColor(temp_img,cv2.COLOR_BGR2RGB))
                    
                    #-----------------------------------------存入json--------------------------------
                    img_id += 1
                    anno_id += 1
                    width,height = temp_img.size
                    anno_result['images'].append({'file_name': save_temp_name,
                                    'id': img_id,
                                    'width': width,
                                    'height': height})
                    x1 = left_top_x
                    y1 = left_top_y
                    x2 = left_top_x+defect_h
                    y2 = left_top_y+defect_w
                    width = max(0, x2 - x1)
                    height = max(0, y2 - y1)
                    anno_result['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': d_name,
                    'id': anno_id,
                    'image_id': img_id,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                    })
                    save_temp_flag = True
                    #---------------------------------------------------------------------------------
                else:
                    continue
    if save_temp_flag:
        temp_img.save(save_dir + save_temp_name)
        print(f'保存图片{save_temp_name}')
        logging.info(f'保存图片{save_temp_name}')
json_name='../../coco_data/annotations/Duck_inject_no_u_all.json'
with open(json_name,'w') as fp:
    json.dump(anno_result, fp, indent = 4, separators=(',', ': ')) 
    print('save success')