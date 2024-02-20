import torch
import torch.nn as nn
from keras.datasets import mnist

class MLP(nn.Module):#이게 다중신경망?!
    def __init__(self):
        super().__init__()
        self.shd=nn.Sequential(#시퀀셜로 묶음
                nn.Linear(28*28,256),#레이어
                nn.Linear(256,10)#다중됨
                )
    def forward(self,x):#아마도 오버라이딩?!
        result01 = self.shd(x)
        return result01


(x_train, y_train), (x_test, y_test) = mnist.load_data()#data samle인듯
x_train=torch.from_numpy(x_train)
x_train=x_train.type(torch.float)#데이터타입이 달라서 float로 조교
flat=nn.Flatten()#그래도 이상해서 납작하게 누름
x_train=flat(x_train)
MJY=MLP()

print(MJY(x_train))


#print(f'x_train={x_train}')
#print(f'y_train={y_train}')
