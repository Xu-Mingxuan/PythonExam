import os
import shutil  # 导入 shutil 模块
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


# 获取 data 文件夹下所有文件夹名（即需要分类的类名）
file_path = 'cat_dog'
flower_class = [cla for cla in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, cla))]

# 创建训练集 train 文件夹，并由类名在其目录下创建子目录
mkfile('data/train')
for cla in flower_class:
    mkfile(os.path.join('data/train', cla))

# 创建验证集 val 文件夹，并由类目在其目录下创建子目录
mkfile('data/test')
for cla in flower_class:
    mkfile(os.path.join('data/test', cla))

# 设置划分比例
split_rate = 0.1

# 遍历所有类别的图像并且按照比例划分为数据集和验证集
for cla in flower_class:
    cla_path = os.path.join(file_path, cla)  # 某一类别的子目录
    images = os.listdir(cla_path)  # images 列表存储了该目录下所有图像的名称
    num = len(images)

    # 从 images 中随机抽取 k 个图像名称
    eval_index = random.sample(images, k=int(num * split_rate))

    for index, image in enumerate(images):
        # eval_index 中保存验证集 val 的图像名称
        if image in eval_index:
            image_path = os.path.join(cla_path, image)
            new_path = os.path.join('data/test', cla)
            shutil.copy(image_path, new_path)  # 使用 shutil.copy
        else:
            # 其余的图像保存在训练集 train 当中
            image_path = os.path.join(cla_path, image)
            new_path = os.path.join('data/train', cla)
            shutil.copy(image_path, new_path)  # 使用 shutil.copy

        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end='')  # processing bar
    print()

print("processing done!")