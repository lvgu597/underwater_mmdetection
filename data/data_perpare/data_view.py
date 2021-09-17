data_source = '../../coco_data/train_data/'

import os 
from lxml import etree
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import shutil
# matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
# matplotlib.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
classes = {}
num = 0
for root,dirs,files in os.walk(f"{data_source}train/box/"):
    for f in files:
        if 'xml' in f:
            file_source = f'{root}{f}'
            page = etree.parse(file_source)
            texts = []
            for t in page.xpath('//name'):
                texts.append(t.text)
            if 'echinus' in texts or 'starfish' in texts or 'holothurian' in texts or 'scallop' in texts:
                continue
            else: #只有海草或者no traget
                print(f)
                shutil.copy(f"{data_source}train/image/{f.replace('xml','jpg')}", '../../coco_data/no_target_images')
                # if classes.get(c,None):
                #     classes[c] += 1
                # else:
                #     classes[c] = 1
# print(classes, num)

# by_value = sorted(classes.items(),key = lambda item:item[1],reverse=True)
# print(by_value)