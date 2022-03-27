from torch import nn
import torch


class Res_net(nn.Module):
    def __init__(self,channel):
        super(Res_net, self).__init__()
        self.res_net = nn.Sequential(
            nn.Conv2d(channel,channel,3,1,padding=1,bias=False),
            nn.BatchNorm2d(channel),
            nn.Dropout2d(0.2),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, 1,padding=1,bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        
    def forward(self,input_x):
        return self.res_net(input_x)+input_x


class Pool(nn.Module):
    def __init__(self,chan_in,chan_out):
        super(Pool, self).__init__()
        self.pool = nn.Sequential(
            nn.Conv2d(chan_in,chan_out,3,1,padding=1,bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(chan_out,chan_out,3,1,padding=1,bias=False),
            nn.BatchNorm2d(chan_out),
            nn.ReLU()
        )

    def forward(self,input_x):
        return self.pool(input_x)


class CNN_net(nn.Module):
    def __init__(self):
        super(CNN_net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Res_net(16),
            Res_net(16),
            Res_net(16),

            Pool(16, 32),
            Res_net(32),
            Res_net(32),
            Res_net(32),

            Pool(32, 64),
            Res_net(64),
            Res_net(64),
            Res_net(64),
            Res_net(64),

            Pool(64,128),
            Res_net(128),
            Res_net(128),
            Res_net(128),
            Res_net(128),
            Res_net(128)
        )
        self.out_net = nn.Sequential(
            nn.Linear(128*6*6, 2)
        )

    def forward(self,input_x):
        cnn_out = self.net(input_x)
        cnn_out = cnn_out.reshape(-1,128*6*6)
        out = self.out_net(cnn_out)
        return out


class Test_net(nn.Module):
    def __init__(self):
        super(Test_net, self).__init__()
        self.test_net = nn.Sequential(
            nn.Conv2d(3,16,3,1),
            nn.MaxPool2d(2)
        )
    def forward(self,x):
        return self.test_net(x)

if __name__ == '__main__':
    a = torch.randn(1,3,100,100)
    net = CNN_net()
    # test_net = Test_net()
    out1 = net(a)
    # out2 = test_net(a)
    print(out1.shape)
    # print(out2.shape)