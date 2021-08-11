import torch
import torch.nn as nn
from PIL import Image
import numpy as np

file_name = '4.png'

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

def image_tva():
    img = Image.open(file_name).convert('L')
    if img.size[0] != 28 or img.size[1] != 28:
        img =img.resize((28,28))
    img_arr = []
    for i in range(28):
        for j in range(28):
            pix = float(img.getpixel((j,i))) / 255.0
            pix = 1.0 - pix
            img_arr.append(pix)
    img = np.array(img_arr).reshape((-1,1,28,28))
    #print(img)
    img = torch.from_numpy(img).float()
    #print(img)
    return(img)

result = image_tva().cuda()
model = CNN().cuda()
model.load_state_dict(torch.load('./model/CNN_NO.pkl'))
model.eval()
test_output = model(result)
pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
print(type(pred_y))
print(pred_y,'prediction number')