from torchvision.datasets import ImageFolder
import copy
import time
from model import AlexNet
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from torch.utils.data import random_split

def train_val_process():
    # 定义数据集路径
    ROOT_TRAIN = r'data\train'

    normalize = transforms.Normalize([0.163,0.153,0.140],[0.058,0.053,0.048])

    # 定义数据集处理方法的变量
    train_transform = transforms.Compose([transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        normalize])

    # 加载数据集
    train_data = ImageFolder(root=ROOT_TRAIN, transform=train_transform)

    # 获取训练数据的长度
    total_size = len(train_data)

    # 计算训练集和验证集的大小
    train_size = round(0.8 * total_size)
    val_size = total_size - train_size  # 余下的样本作为验证集

    train_data, val_data = random_split(train_data, [train_size, val_size])

    train_loader = DataLoader(dataset=train_data,
                              batch_size=64,
                              shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(dataset=val_data,
                            batch_size=64,
                            shuffle=True,
                            num_workers=0)
    return train_loader, val_loader

def train_model_process(model, train_loader, val_loader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0

        # 训练过程
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.train()

            outputs = model(b_x)
            pre_lab = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        # 验证过程
        model.eval()
        with torch.no_grad():
            for step, (b_x, b_y) in enumerate(val_loader):
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                outputs = model(b_x)
                pre_label = torch.argmax(outputs, dim=1)

                loss = criterion(outputs, b_y)
                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_label == b_y.data)
                val_num += b_x.size(0)

        # 更新损失和准确度
        train_loss_all.append(train_loss / train_num if train_num > 0 else 0)
        val_loss_all.append(val_loss / val_num if val_num > 0 else 0)
        train_acc_all.append(train_corrects.double() / train_num if train_num > 0 else 0)
        val_acc_all.append(val_corrects.double() / val_num if val_num > 0 else 0)

        print("{} Train Loss: {:.4f} train ACC: {:.4f}".format(epoch + 1, train_loss_all[-1], train_acc_all[-1]))
        print("{} val Loss: {:.4f} val ACC: {:.4f}".format(epoch + 1, val_loss_all[-1], val_acc_all[-1]))

        # 保存模型参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        # 打印列表长度
        print(len(train_loss_all), len(train_acc_all), len(val_loss_all), len(val_acc_all))

        # 计算耗时
        time_use = time.time() - since
        print("训练耗费时间：{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    # 加载最佳模型参数
    torch.save(best_model_wts, 'Alexnet.pth')

    train_loss_all = [x.cpu() if isinstance(x, torch.Tensor) else x for x in train_loss_all]
    val_loss_all = [x.cpu() if isinstance(x, torch.Tensor) else x for x in val_loss_all]
    train_acc_all = [x.cpu() if isinstance(x, torch.Tensor) else x for x in train_acc_all]
    val_acc_all = [x.cpu() if isinstance(x, torch.Tensor) else x for x in val_acc_all]

    # 保存各个数值到 pandas 的 DataFrame 中
    train_process = pd.DataFrame                                                (data={
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "train_acc_all": train_acc_all,
        "val_loss_all": val_loss_all,
        "val_acc_all": val_acc_all
    })

    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, "ro-", label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, "bs-", label="val loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, "ro-", label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, "bs-", label="val acc")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.show()


if __name__ == '__main__':
    # 实例化模型
    AlexNet = AlexNet()
    # 加载数据集
    train_loader, val_loader = train_val_process()
    # 得到最后的loss值用来画图
    train_process = train_model_process(AlexNet,train_loader,val_loader,20)

    matplot_acc_loss(train_process)


















