#导入transforms函数对图像做预处理
from torchvision import transforms

'''
Compose()将下列操作整合到一起：
将torch.tensor 转换为PIL图像
图像尺寸转换到给定尺寸
以0.5概率水平翻转给定的PIL图像
将PIL图像转换为tensor
用均值[0.5,0.5,0.5]和标准差[0.5,0.5,0.5]对图像做归一化处理
'''
def transform_for_training(image_shape):
    return transforms.Compose(
       [transforms.ToPILImage(),
        transforms.Resize(image_shape),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )


def transform_for_infer(image_shape):
    return transforms.Compose(
       [transforms.ToPILImage(),
        transforms.Resize(image_shape),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
