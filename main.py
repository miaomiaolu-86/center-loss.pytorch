import os
#导入命令行参数标准库
import argparse
#导入pytorch各种功能和工具
import torch   
'''
批训练，把数据变成一小批一小批数据进行训练。
DataLoader用来包装所使用的数据，每次抛出一批数据
'''
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from dataset import Dataset, create_datasets, LFWPairedDataset
from loss import compute_center_loss, get_center_delta
from models import Resnet50FaceModel, Resnet18FaceModel
from device import device
from trainer import Trainer
from utils import download, generate_roc_curve, image_loader
from metrics import compute_roc, select_threshold
from imageaug import transform_for_infer, transform_for_training


def main(args):
    if args.evaluate:
        evaluate(args)
    elif args.verify_model:
        verify(args)
    else:
        train(args)

#获取数据集目录
def get_dataset_dir(args):
    #将路径字符串中的波浪线（~）扩展为用户的主目录，提供跨平台路径展开
    home = os.path.expanduser("~")
    #获取数据集路径
    dataset_dir = args.dataset_dir if args.dataset_dir else os.path.join(
        home, 'datasets', 'lfw')
#判断路径是否是目录，不是目录的话，创造一个目录
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    return dataset_dir


def get_log_dir(args):
    #去掉文件名，返回目录
    log_dir = args.log_dir if args.log_dir else os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'logs')

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    return log_dir

#根据参数量，选择模型训练使用的网络类型
def get_model_class(args):
    if args.arch == 'resnet18':
        model_class = Resnet18FaceModel
    if args.arch == 'resnet50':
        model_class = Resnet50FaceModel
    elif args.arch == 'inceptionv3':
        model_class = InceptionFaceModel

    return model_class

'''
训练
'''
def train(args):
    #初始化变量
    dataset_dir = get_dataset_dir(args)
    log_dir = get_log_dir(args)
    model_class = get_model_class(args)

    training_set, validation_set, num_classes = create_datasets(dataset_dir)

    training_dataset = Dataset(
            training_set, transform_for_training(model_class.IMAGE_SHAPE))
    validation_dataset = Dataset(
        validation_set, transform_for_infer(model_class.IMAGE_SHAPE))

    #加载训练数据，每一批图片打乱顺序
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=True
    )

    #加载验证数据，无需打乱顺序
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=False
    )

    #选择运行设备
    model = model_class(num_classes).to(device)

    #设置梯度？？？
    trainables_wo_bn = [param for name, param in model.named_parameters() if
                        param.requires_grad and 'bn' not in name]
    trainables_only_bn = [param for name, param in model.named_parameters() if
                          param.requires_grad and 'bn' in name]

    '''
    构建一个优化器
    momentum=0.9，动量加速优化'''
    optimizer = torch.optim.SGD([
        {'params': trainables_wo_bn, 'weight_decay': 0.0001},
        {'params': trainables_only_bn}
    ], lr=args.lr, momentum=0.9)

    #进行训练
    trainer = Trainer(
        optimizer,
        model,
        training_dataloader,
        validation_dataloader,
        max_epoch=args.epochs,
        resume=args.resume,
        log_dir=log_dir
    )
    trainer.train()

'''
测试
'''
def evaluate(args):
    dataset_dir = get_dataset_dir(args)
    log_dir = get_log_dir(args)
    model_class = get_model_class(args)

    '''拼接目录，读取pairs.txt文件
    如无文件，则下载一个文件
    '''
    pairs_path = args.pairs if args.pairs else \
        os.path.join(dataset_dir, 'pairs.txt')

    if not os.path.isfile(pairs_path):
        download(dataset_dir, 'http://vis-www.cs.umass.edu/lfw/pairs.txt')

    #加载数据集
    dataset = LFWPairedDataset(
        dataset_dir, pairs_path, transform_for_infer(model_class.IMAGE_SHAPE))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    model = model_class(False).to(device)

    #加载模型参数
    checkpoint = torch.load(args.evaluate)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    #不更新当前模型参数
    model.eval()

    #设置全0张量
    embedings_a = torch.zeros(len(dataset), model.FEATURE_DIM)
    embedings_b = torch.zeros(len(dataset), model.FEATURE_DIM)
    matches = torch.zeros(len(dataset), dtype=torch.uint8)

    for iteration, (images_a, images_b, batched_matches) \
            in enumerate(dataloader):
        current_batch_size = len(batched_matches)
        images_a = images_a.to(device)
        images_b = images_b.to(device)

        _, batched_embedings_a = model(images_a)
        _, batched_embedings_b = model(images_b)

        start = args.batch_size * iteration
        end = start + current_batch_size

        embedings_a[start:end, :] = batched_embedings_a.data
        embedings_b[start:end, :] = batched_embedings_b.data
        matches[start:end] = batched_matches.data

    thresholds = np.arange(0, 4, 0.1)
    '''
    计算两个张量差值的平方
    对张量按行求和
    '''
    distances = torch.sum(torch.pow(embedings_a - embedings_b, 2), dim=1)

    tpr, fpr, accuracy, best_thresholds = compute_roc(
        distances,
        matches,
        thresholds
    )

    roc_file = args.roc if args.roc else os.path.join(log_dir, 'roc.png')
    generate_roc_curve(fpr, tpr, roc_file)
    print('Model accuracy is {}'.format(accuracy))
    print('ROC curve generated at {}'.format(roc_file))

'''
验证
'''
def verify(args):
    dataset_dir = get_dataset_dir(args)
    log_dir = get_log_dir(args)
    model_class = get_model_class(args)

    model = model_class(False).to(device)
    #加载模型参数
    checkpoint = torch.load(args.verify_model)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    #不更新模型参数
    model.eval()

    image_a, image_b = args.verify_images.split(',')
    image_a = transform_for_infer(
        model_class.IMAGE_SHAPE)(image_loader(image_a))
    image_b = transform_for_infer(
        model_class.IMAGE_SHAPE)(image_loader(image_b))
    images = torch.stack([image_a, image_b]).to(device)

    _, (embedings_a, embedings_b) = model(images)

    distance = torch.sum(torch.pow(embedings_a - embedings_b, 2)).item()
    print("distance: {}".format(distance))


if __name__ == '__main__':

    #创建解析器
    parser = argparse.ArgumentParser(description='center loss example')
    #添加参数
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--log_dir', type=str,
                        help='log directory')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--arch', type=str, default='resnet50',
                        help='network arch to use, support resnet18 and '
                             'resnet50 (default: resnet50)')
    parser.add_argument('--resume', type=str,
                        help='model path to the resume training',
                        default=False)
    parser.add_argument('--dataset_dir', type=str,
                        help='directory with lfw dataset'
                             ' (default: $HOME/datasets/lfw)')
    parser.add_argument('--weights', type=str,
                        help='pretrained weights to load '
                             'default: ($LOG_DIR/resnet18.pth)')
    parser.add_argument('--evaluate', type=str,
                        help='evaluate specified model on lfw dataset')
    parser.add_argument('--pairs', type=str,
                        help='path of pairs.txt '
                             '(default: $DATASET_DIR/pairs.txt)')
    parser.add_argument('--roc', type=str,
                        help='path of roc.png to generated '
                             '(default: $DATASET_DIR/roc.png)')
    parser.add_argument('--verify-model', type=str,
                        help='verify 2 images of face belong to one person,'
                             'the param is the model to use')
    parser.add_argument('--verify-images', type=str,
                        help='verify 2 images of face belong to one person,'
                             'split image pathes by comma')

    #解析参数
    args = parser.parse_args()
    main(args)
