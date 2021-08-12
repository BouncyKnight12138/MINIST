import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 2 #训练次数，似乎1就够用了
BATCH_SIZE = 50 #张量中的图片数量
LR = 0.001# 学习率
DOWNLOAD_MNIST = True

#训练集数据
train_data = torchvision.datasets.MNIST(
    root='./mnist', #保存位置
    train=True, #是否为训练集
    transform=torchvision.transforms.ToTensor(), #转化为tensor数据格式
    download=DOWNLOAD_MNIST,
)
#测试集数据
test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
)
#加载训练集数据，在训练集张量中有50个图片
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

#将数据形式由(2000, 28, 28) 转化为 (2000, 1, 28, 28)，且范围为[0,1]
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000]/255.
#取前2000个的标签
test_y = test_data.targets[:2000]
test_x = test_x.cuda()
test_y = test_y.cuda()

# 卷积层使用 torch.nn.Conv2d
# 激活层使用 torch.nn.ReLU
# 池化层使用 torch.nn.MaxPool2d
# 全连接层使用 torch.nn.Linear

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,#移动步数，左移和下移都是一样的，如果是2则左移2，完成后下移也是2
                padding=2,#填充大小
            ),#由于输出通道是16，并且填充了2，所以输出结果为(16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#在2*2空间向下采样进行池化(16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),#out(32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),#out(32,7,7)
        )
        self.out = nn.Linear(32*7*7 , 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)#展平多维的卷积图成 x.size(0)是batch的大小，所以x = x.view(x.size(0), -1)相当于x = x.view(BATCH_SIZE, -1)。
                                 #-1是指列数未知的情况下，根据原来Tensor的数据和BATCH_SIZE自动分配列数
        output = self.out(x)#全连接
        return output

cnn = CNN()
cnn = cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)#使用adam优化算法
loss_func = nn.CrossEntropyLoss()#交叉熵损失函数
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.cuda()#图片
        b_y = b_y.cuda()#标签
        output = cnn(b_x)#cnn output, the size of b_x is ([batchsize, channel, height, width)
        loss = loss_func(output, b_y)#cross entropy loss
        optimizer.zero_grad()#clear gradients for this training step 将梯度归零
        loss.backward()#反向传播求梯度
        optimizer.step()#通过梯度做一步参数更新
        if step %50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            accuracy = float((pred_y == test_y).sum()) / float(test_y.size(0))
            print('epoch:', epoch,'| train_loss: %.4f' %loss.data,'|test accuracy %.2f'%accuracy)


test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')

torch.save(cnn.state_dict(), './model/CNN_NO.pkl')
