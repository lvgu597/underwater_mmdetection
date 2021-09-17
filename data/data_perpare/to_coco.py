import json
import os
import cv2
import xml.etree.ElementTree as ET
import shutil
from lxml import etree
def parse_xml(xml_path):
    tree = ET.parse(xml_path)		
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(box[0].text)
        y_min = int(box[1].text)
        x_max = int(box[2].text)
        y_max = int(box[3].text)
        coords.append([x_min, y_min, x_max, y_max, name])
    return coords

def convert(root_path, source_xml_root_path, target_xml_root_path, phase='train', split=270): #这里split设置有问题
    dataset = {'categories':[], 'images':[], 'annotations':[]}
    # classes = {'echinus': 1, 'starfish': 2, 'holothurian': 3, 'scallop': 4}
    classes = {'Port': 1, 'Airport':2}
    for i, cls in enumerate(classes, 1):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'Geography'})   #mark
    
#     # 读取images文件夹的图片名称
    pics = []
    file_list = os.listdir(os.path.join(root_path, 'Annotations'))
    for file in file_list:
        file_source = f"{source_xml_root_path}{file}"
        # print(file_source)
        page = etree.parse(file_source)
        texts = []
        for t in page.xpath('//name'):
            texts.append(t.text)
        if 'Port' in texts or 'Airport' in texts :
            pics.append(file.replace('xml','tif'))
        # elif 'waterweeds' in texts:
        #     print(file_source)
    print(len(pics))
    # # 移动文件操作
    # for file in pics:
    #     shutil.copy(f'../../coco_data/train_data/train/image/{file}',f'../../coco_data/defect_images/')
    #     #print(file)
    
    #　pics = [f for f in os.listdir(os.path.join(root_path, 'data_img'))]
    # 判断是建立训练集还是验证集
    if phase == 'train':
        pics = [line for i, line in enumerate(pics) if i <= split]
    elif phase == 'val':
        pics = [line for i, line in enumerate(pics) if i > split]
    
    print('---------------- start convert ---------------')
    bnd_id = 1	#初始为1
    for i, pic in enumerate(pics):
        xml_path = os.path.join(source_xml_root_path, pic[:-4]+'.xml')
        pic_path = os.path.join(root_path, 'defect_images/' + pic)
        print(pic_path)
        im = cv2.imread(pic_path)
        height, width, _ = im.shape
        dataset['images'].append({'file_name': pic,
                                  'id': i,
                                  'width': width,
                                  'height': height})
        try:
            coords = parse_xml(xml_path)
        except:
            print(pic[:-4]+'.xml not exists~')
            continue
        for coord in coords:
            # x_min
            x1 = int(coord[0])-1
            x1 = max(x1, 0)
            # y_min
            y1 = int(coord[1])-1
            y1 = max(y1, 0)
            # x_max
            x2 = int(coord[2])
            # y_max
            y2 = int(coord[3])
            assert x1<x2
            assert y1<y2
            # name
            name = coord[4]
            if classes.get(name):
                cls_id = classes.get(name)
            else:
                continue
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            dataset['annotations'].append({
                'area': width * height,
                'bbox': [x1, y1, width, height],
                'category_id': cls_id,
                'id': bnd_id,
                'image_id': i,
                'iscrowd': 0,
                # mask, 矩形是从左上角点按顺时针的四个顶点
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            })
            bnd_id += 1
            
    # 保存结果的文件夹
#     folder = os.path.join(target_xml_root_path, 'annotations')
#     if os.path.exists(folder):
#         shutil.rmtree(folder)
#     os.makedirs(folder)
    json_name = os.path.join(target_xml_root_path, 'instances_{}0702.json'.format(phase))
    with open(json_name, 'w') as f:
        json.dump(dataset, f, indent = 4, separators=(',', ': '))
        print('save success')

if __name__ == '__main__':
    convert(root_path='../../coco_data/train/', 
            source_xml_root_path = '../../coco_data/train/Annotations/', 
            target_xml_root_path = '../../coco_data/train/annotations/',phase='val')