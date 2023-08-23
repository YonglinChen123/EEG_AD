import torch
from torch import nn
# 导入记好了，         2维卷积，2维最大池化，展成1维，全连接层，构建网络结构辅助工具,2d网络归一化,激活函数,自适应平均池化
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d
from torchsummary import summary
import cv2
import time
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
##########################################
def draw_features(width, height, x, savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    import matplotlib.image as img
    for i in range(width * height):
        ax = plt.subplot(height, width, i + 1)
        plt.axis('off')

        #img = x[0, i, :, :]
        import matplotlib.image as img
        #img = x[0, i, :, :]
        img = x
        # ix = np.unravel_index(i, ax)
        # img = x[ix]

        pmin = np.min(img)
        pmax = np.max(img)
        #img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        #plt.matshow(img[:, :, ::-1], cmap=plt.get_cmap('gray'))
        #plt.matshow(img((15, 15)), cmap='viridis')
        #plt.imshow(img)
        #cm1 = plt.cm.get_cmap('jet')
        #img = img.imread('1003.jpg')
        # plt.colorbar()
        # print(set(img.flatten())) # {0.007843138, 0.011764706, 0.003921569} 和{1，2，3}
        plt.imshow(img)
        #plt.colorbar()
        #plt.matshow(img)
        #plt.colorbar()
        print("{}/{}".format(i, width * height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))
##########################################

class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()
        self.model0 = Sequential(
            # 0
            # 输入3通道、输出64通道、卷积核大小、步长、补零、
            #Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 维度展平
        self.flatten = Flatten(start_dim=1)
        # FC 全连接层
        #self.fc = Linear(512, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        x = self.model0(x)

        #savepath = r'features_whitegirl'
        #draw_features(1, 1, x.detach().cpu().numpy(), "{}/f1_conv1.png".format(savepath))
        f1 = x
        x = self.model1(x)
        #draw_features(1, 1, x.detach().cpu().numpy(), "{}/f1_conv2.png".format(savepath))
        x = x + f1
        x = self.R1(x)

        f1_1 = x
        x = self.model2(x)
        #draw_features(1, 1, x.detach().cpu().numpy(), "{}/f1_conv3.png".format(savepath))
        x = x + f1_1
        x = self.R2(x)

        f2_1 = x
        f2_1 = self.en1(f2_1)
        x = self.model3(x)
        #draw_features(1, 1, x.detach().cpu().numpy(), "{}/f1_conv4.png".format(savepath))
        x = x + f2_1
        x = self.R3(x)

        f2_2 = x
        x = self.model4(x)
        #draw_features(1, 1, x.detach().cpu().numpy(), "{}/f1_conv5.png".format(savepath))
        x = x + f2_2
        x = self.R4(x)

        f3_1 = x
        f3_1 = self.en2(f3_1)
        x = self.model5(x)
        #draw_features(1, 1, x.detach().cpu().numpy(), "{}/f1_conv5.png".format(savepath))
        x = x + f3_1
        x = self.R5(x)

        f3_2 = x
        x = self.model6(x)
        #draw_features(1, 1, x.detach().cpu().numpy(), "{}/f1_conv6.png".format(savepath))
        x = x + f3_2
        x = self.R6(x)

        f4_1 = x
        f4_1 = self.en3(f4_1)
        x = self.model7(x)
        #draw_features(1, 1, x.detach().cpu().numpy(), "{}/f1_conv7.png".format(savepath))
        x = x + f4_1
        x = self.R7(x)

        f4_2 = x
        x = self.model8(x)
        #draw_features(1, 1, x.detach().cpu().numpy(), "{}/f1_conv9.png".format(savepath))
        x = x + f4_2
        x = self.R8(x)

        # 最后3个
        x = self.aap(x)
        x = self.flatten(x)

        x = self.fc(x)

        #draw_features(1, 1, x.detach().cpu().numpy(), "{}/f1_conv10.png".format(savepath))
        # print(888)
        # print(x.shape)
        # print(888)
        return x


if __name__ == '__main__':
    # 3分类
    res18 = Resnet18(3).to('cuda:0')
    summary(res18, (3, 224, 224))
