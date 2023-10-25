#Python标准库os库，包含几百个函数,常用路径操作、进程管理、环境参数等几类
import os
from math import ceil

from tqdm import tqdm
import numpy as np
import cv2
import requests
import matplotlib.pyplot as plt


def download(dir, url, dist=None):
    dist = dist if dist else url.split('/')[-1]
    print('Start to Download {} to {} from {}'.format(dist, dir, url))
    
    '''os.path子库，处理文件路径及信息
    获得文件路径'''
    download_path = os.path.join(dir, dist)
    if os.path.isfile(download_path):
        print('File {} already downloaded'.format(download_path))
        return download_path
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 * 1024

    with open(download_path, 'wb') as f:
        for data in tqdm(
                r.iter_content(block_size),
                total=ceil(total_size//block_size),
                unit='MB', unit_scale=True):
            f.write(data)
    print('Downloaded {}'.format(dist))
    return download_path


def image_loader(image_path):
    return cv2.imread(image_path)


def generate_roc_curve(fpr, tpr, path):
    assert len(fpr) == len(tpr)

    fig = plt.figure()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(fpr, tpr)
    fig.savefig(path, dpi=fig.dpi)
