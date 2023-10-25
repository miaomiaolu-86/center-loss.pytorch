#Python标准库os库，包含几百个函数,常用路径操作、进程管理、环境参数等几类
import os
from math import ceil
#导入进度条库
from tqdm import tqdm
import numpy as np
import cv2
import requests
import matplotlib.pyplot as plt

#获取文件路径
def download(dir, url, dist=None):
    #获取url最后一个/后的字符串
    dist = dist if dist else url.split('/')[-1]
    
    '''format()控制字符串和变量显示方式
    输出：从url地址开始下载字符串到目录dir下面'''
    print('Start to Download {} to {} from {}'.format(dist, dir, url))
    
    '''os.path子库，处理文件路径及信息
    获得文件路径'''
    download_path = os.path.join(dir, dist)
    
    #判断该路径下对象是否是文件
    if os.path.isfile(download_path):
        print('File {} already downloaded'.format(download_path))
        return download_path
        
    #若目录下不是文件，获取网页的内容
    r = requests.get(url, stream=True)
    #爬取数据，获得内容长度，
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 * 1024

    #以写二进制的方式打开该路径下的文件
    with open(download_path, 'wb') as f:
        for data in tqdm(
            #迭代读取文件，每次读取1024 * 1024大小的数据块    
            r.iter_content(block_size),
            #向上取整
            total=ceil(total_size//block_size),
                unit='MB', unit_scale=True):
            #信息写入到文件        
            f.write(data)
    print('Downloaded {}'.format(dist))
    return download_path

#读取图片
def image_loader(image_path):
    return cv2.imread(image_path)

#画图
def generate_roc_curve(fpr, tpr, path):
    #判断长度是否是0，是0触发异常
    assert len(fpr) == len(tpr)

    #创建一个图形
    fig = plt.figure()
    #设置X、Y轴标签
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    #画图
    plt.plot(fpr, tpr)
    #将图形保存为文件
    fig.savefig(path, dpi=fig.dpi)
