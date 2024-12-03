from torchvision.datasets import ImageFolder
import copy
import datetime as time
from model import AlexNet
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.utils.data as Data
import pandas as pd


def train_val_process():
    # 定义数据集路径
    ROOT_TRAIN = r'data\train'

    normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.05,0.052,0.047])

    # 定义数据集处理方法的变量
    train_transform = transforms.Compose([transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        normalize])

    # 加载数据集
    train_data = ImageFolder(root=ROOT_TRAIN, transform=train_transform)

    train_data, val_data = Data.random_split(train_data,[round(0.8*(train_data))],[round(0.2*(train_data))])

    train_loader = DataLoader(dataset=train_data,
                              batch_size=64,
                              shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(dataset=val_data,
                            batch_size=64,
                            shuffle=True,
                            num_workers=0)
    return train_loader, val_loader

def train_model_process(model,train_loader,val_loader,num_epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 梯度下降
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 将模型放到训练设备当中
    model = model.to(device)

    # 复制当前模型的参数，用于后面测试时加载
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0

    # 训练集损失值列表
    train_loss_all = []

    # 验证集损失值列表
    val_loss_all = []

    # 训练集准确度列表
    train_acc_all = []

    # 测试集准确度列表
    val_acc_all = []

    # 当前时间
    since=time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)


        # 初始化参数值，损失值，准确度
        train_loss = 0.0
        train_corrects = 0

        # 验证集损失值，准确度
        val_loss = 0.0
        val_corrects = 0

        # 验证集，训练集的样本数量
        train_num = 0
        val_num = 0

        # 读取数据
        for step, (b_x, b_y) in enumerate(train_loader):
            # 将特征放到和模型一样的设备当中
            # 数据
            b_x = b_x.to(device)
            # 标签
            b_y = b_y.to(device)

            # 打开模型的训练模式
            model.train()
            # 前向传播，输入一个batch，输出一个batch中的预测
            outputs = model(b_x)

            # 查找最大值的行标，就是概率最高的类别
            pre_lab = torch.argmax(outputs, dim=1)

            # 计算损失值
            loss = criterion(pre_lab, b_y)

            # 将梯度初始化为0,避免之前的梯度累加进行干扰,然后反向传播
            optimizer.zero_grad()
            loss.backward()

            # 进行参数更新
            optimizer.step()

            # 对损失函数进行累加，用来评估一个epoch的表现
            train_loss += loss.item() * b_x.size()

            # 精确度累加,b_y是标签，.data是取值
            train_corrects += torch.sum(pre_lab == b_y.data)

            # 获取训练集样本数量
            train_num += b_x.size


        for step, (b_x, b_y) in enumerate(val_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()

            outputs = model(b_x)
            pre_label = torch.argmax(outputs, dim=1)


            loss = criterion(pre_label, b_y)

            val_loss += loss.item() * b_x.size()

            val_corrects += torch.sum(b_y == pre_label.data)

            val_num += b_x.size


        # 总损失值
        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)

        # 总正确率
        train_acc_all.append(train_corrects.double() / train_num)
        val_acc_all.append(val_corrects.double() / val_num)

        print("{} Train Loss: {:.4f} train ACC: {:.4f}".format(epoch+1, train_loss_all[-1],train_acc_all[-1]))
        print("{} val Loss: {:.4f} val ACC: {:.4f}".format(epoch+1, val_loss_all[-1],val_acc_all[-1]))


        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]
            # 保存模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

            # 保存各个数值到pandas的DataFrame当中
            train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                               "train_loss_all": train_loss_all,
                                               "train_acc_all": train_acc_all,
                                               "val_loss_all": val_loss_all,
                                               "val_acc_all": val_acc_all})
        # 耗时
        time_use = time.time() - since
        print("训练耗费时间：{:.0f}m{:.0f}s".format(time_use//60, time_use%60))

    # 加载最高准确率下的模型参数
    torch.save(best_model_wts, 'lenet.pth')


    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, "ro-", label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, "bs-", label="val loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, "ro-", label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, "bs-", label="val acc")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()


if __name__ == '__main__':
    # 实例化模型
    AlexNet = AlexNet()
    # 加载数据集
    train_loader, val_loader = train_val_process()
    # 得到最后的loss值用来画图
    train_process = train_model_process(AlexNet,train_loader,val_loader,20)

    matplot_acc_loss(train_process)


















