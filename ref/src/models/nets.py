import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class SimpleCNN(nn.Module):
    def __init__(self,n=5):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,n))
    def forward(self,x): return self.net(x)
def block(a,b,p=True):
    z=[nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(inplace=True),nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(inplace=True)]
    return z+([nn.MaxPool2d(2)] if p else [])
class ComplexCNN(nn.Module):
    def __init__(self,n=5):
        super().__init__(); self.f=nn.Sequential(*block(3,32),*block(32,64),*block(64,128),*block(128,256,False),nn.AdaptiveAvgPool2d(1)); self.h=nn.Sequential(nn.Flatten(),nn.Dropout(.35),nn.Linear(256,n))
    def forward(self,x):return self.h(self.f(x))
def mobile(pretrained=True,n=5):
    m=mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None);m.classifier[1]=nn.Linear(m.last_channel,n);return m
