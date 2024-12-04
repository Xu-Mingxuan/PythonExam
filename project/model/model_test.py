import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST, ImageFolder
from model import AlexNet
from PIL import Image


def test_data_process():
    ROOT_TEST = r'data\test'

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.05, 0.052, 0.047])

    # 定义数据集处理方法的变量
    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          normalize])

    test_data = ImageFolder(root = ROOT_TEST,
                              transform = test_transform)

    test_loader = DataLoader(dataset=test_data,
                              batch_size=64,
                              shuffle=True,
                              num_workers=0)

    return test_loader

def test_model_process(model,test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将模型放到设备当中
    model = model.to(device)

    # 初始化一些参数
    test_cor = 0

    test_num = 0

    # 声明只进行前向传播，不进行反向传播
    with torch.no_grad():
        for test_data_x, test_data_y in test_loader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            # 将模型设置为验证模式
            model.eval()
            # 前向传播
            output = model(test_data_x)

            x = torch.argmax(output, dim=1)

            test_cor += torch.sum(x == test_data_y.data)

            test_num += test_data_x.size(0)

        test_acc = test_cor / test_num
        print("测试的准确率为：{:.4f}".format(test_acc))


if __name__ == '__main__':
    # 加载模型和模型参数
    model = AlexNet()
    model.load_state_dict(torch.load('Alexnet.pth'))

    # 加载数据集
    test_loader = test_data_process()

    # # 加载模型测试函数
    # test_model_process(model,test_loader)
    #
    #
    # # 验证模式
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    #
    #
    #
    classes = ["猫","狗"]
    with torch.no_grad():

        image = Image.open('2.jpg')

        normalize = transforms.Normalize([0.163,0.153,0.140],[0.058,0.053,0.048])

        test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize])
        image = test_transform(image)

        # 添加一个批次为一的维数，符合输入条件
        image = image.unsqueeze(0)

        with torch.no_grad():
            model.eval()
            image = image.to(device)
            output = model(image)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
        print('预测值为： ',classes[result])