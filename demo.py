import torch
import torch.nn as nn
from PIL import Image
import numpy as np

file_name = '6.png'#识别图片地址

#模型与训练一样
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32*7*7 , 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
#图像处理
def image_tva():
    img = Image.open(file_name).convert('L')#灰度打开
    if img.size[0] != 28 or img.size[1] != 28:
        img =img.resize((28,28))#调整图像大小到(28*28)
    img_arr = []
    for i in range(28):
        for j in range(28):
            pix = float(img.getpixel((j,i))) / 255.0
            pix = 1.0 - pix#格式化成minist数据  这里的0 代表的是黑，1 代表白，但是minist数据0代表白，1代表黑
            img_arr.append(pix)
    img = np.array(img_arr).reshape((-1,1,28,28))#整理成为(1,1,28,28)
    #print(img.shape)
    img = torch.from_numpy(img).float()
    #print(img)
    return(img)

result = image_tva().cuda()
model = CNN().cuda()
model.load_state_dict(torch.load('./model/CNN_NO.pkl'))#提取参数
model.eval()#使用eval函数展开model
test_output = model(result)
pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
#print(type(pred_y))
print('prediction number:',pred_y.item())