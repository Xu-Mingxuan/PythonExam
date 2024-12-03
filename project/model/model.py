import torch
from torch import nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.relu =nn.ReLU()
        self.c1 = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0)
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c3 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.s4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c5 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.c6 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.c7 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.s8 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(256 * 6 * 6, 4096)
        self.f2 = nn.Linear(4096, 4096)
        self.f3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.c1(x)
        x = self.relu(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.relu(x)
        x = self.s4(x)
        x = self.c5(x)
        x = self.relu(x)
        x = self.relu(self.c6(x))
        x = self.relu(self.c7(x))
        x = self.s8(x)

        x = self.flatten(x)
        x = self.relu(self.f1(x))
        x = F.dropout(x, 0.5)
        x = self.relu(self.f2(x))
        x = F.dropout(x, 0.5)
        x = self.f3(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet()
    model = model.to(device)
    print(summary(model, (1, 227, 227)))