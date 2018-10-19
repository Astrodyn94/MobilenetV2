import torch
import torch.nn as nn
from torch.autograd import Variable


class bottleneck(nn.Module):
    def __init__(self, input_c,t,out,n,s,linear=True,connect = True,cuda = True):
        super(bottleneck, self).__init__()
        self.linear = linear
        self.cuda = cuda
        self.connect = connect #Default; whether you want to use connection or not?
        self.n = n
        self.stride = s
        self.out = out # of output channel
        self.t = t # expanding ratio
        self.input_c = input_c # number of input channel
        self.layer = nn.Sequential(
        nn.Conv2d(self.input_c, self.input_c * t, 1, 1, 0, bias=False),
        nn.BatchNorm2d(self.input_c * t),
        nn.ReLU6(inplace=True),
        #DEPTHWISE
        nn.Conv2d(self.input_c * t, self.input_c * t, 3, self.stride, 1, groups=self.input_c * self.t, bias=False), #groups indicates depthwise convolution
        nn.BatchNorm2d(self.input_c * self.t),
        nn.ReLU6(inplace=True),
        #POINTWISE
        nn.Conv2d(self.input_c *self.t , self.out ,1 ,1,0,bias = False),
        nn.BatchNorm2d(self.out)
        )
        
        if self.connect: 
            if self.out/self.stride**2 == input_c:
                self.connect = True
            else:
                self.connect = False
                
        if linear == False:
            self.layer.add_module('non_linear_bottleneck' , nn.ReLU6(inplace = True))
        if cuda:
            self.layer.cuda()

    def forward(self, x):
        h = x
        for i in range(self.n):
            if i != 0:
                self.stride = 1
            if self.connect:
                self.__init__(self.input_c,self.t,self.out,self.n,self.stride,self.linear,self.connect,self.cuda)
                h = h + self.layer(h)
            else:
                self.__init__(self.input_c,self.t,self.out,self.n,self.stride,self.linear,self.connect,self.cuda)
                h = self.layer(h)
            self.input_c = self.out
        return h


class MobileNetV2(nn.Module):
    def __init__(self, opt):
        super(MobileNetV2, self).__init__()
        self.input_c = opt.input_c
        self.final_c = opt.final_c
        input_c = self.input_c; final_c = self.final_c

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, input_c, 3,2,1,bias = False),
            nn.BatchNorm2d(input_c),
            nn.ReLU6(inplace = True)
        )
        
        self.bottleneck_layers =[]
        for t,c,n,s in opt.architecture:
            self.bottleneck_layers.append(bottleneck(input_c,t,c,n,s,linear=opt.linear,connect = opt.connect,cuda = opt.use_gpu))
            input_c = c
        self.bottleneck_layers = nn.Sequential(*self.bottleneck_layers)
        
        self.last_layer = nn.Sequential(
            nn.Conv2d(input_c,final_c,1,1,0,bias = False),
            nn.BatchNorm2d(final_c),
            nn.ReLU6(inplace = True)
        )
        self.avgpool = nn.AvgPool2d(7 , stride = 1)
        self.classifier = nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(final_c, opt.n_class),
        )
        self.initial_params()


    def forward(self, x):
        x = self.input_layer(x)
        x = self.bottleneck_layers(x)  
        x = self.last_layer(x)
        x = self.avgpool(x)
        x = self.classifier(x.view(-1 , self.final_c))
        return x

    def initial_params(self):
        for m in self.modules():
            if isinstance(m , nn.Conv2d):
                nn.init.kaiming_normal_(m.weight , mode = 'fan_out' , nonlinearity = 'relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight , 1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
