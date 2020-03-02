## HOT idea: add Prior Bayes in neural network
import torch
from torch import nn

C = 3 # input channel

# the equivalent output structure with level 1 scattering transform
class ModelLevel1(nn.Module):
    def __init__(self):
        super(ModelLevel1, self).__init__()
        self.conv1 = nn.Conv2d(1,9,5, padding=(2,2))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        
    def forward(self, x):
        out = x.view(-1, 1, x.shape[2], x.shape[3])
        out = self.pool1(self.relu1(self.conv1(out)))
        out = out.view(-1,9*C,out.shape[2], out.shape[3])
        return out

# remove MaxPool Layer    
class ModelLevel1_I1(nn.Module):
    def __init__(self):
        super(ModelLevel1_I1, self).__init__()
        self.conv1 = nn.Conv2d(1,9,5,padding=2, stride=2)
        self.relu1 = nn.ReLU()
        
    def forward(self, x):
        out = x.view(-1, 1, x.shape[2], x.shape[3])
        out = self.relu1(self.conv1(out))
        out = out.view(-1, 9*C, out.shape[2], out.shape[3])
        return out

# create 2 component for real and imag part
class ModelLevel1_Learn(nn.Module):
    def __init__(self):
        super(ModelLevel1_Learn, self).__init__()
        
        self.conv1_lowpass = nn.Conv2d(1,1,5,padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv1_real = nn.Conv2d(1,8,5,padding=2)        
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv1_imag = nn.Conv2d(1,8,5,padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        
        self.eps = 1e-4

    def forward(self, x):
        out = x.view(-1, 1, x.shape[2], x.shape[3])
        
        out_lowpass = self.pool1(self.relu1(self.conv1_lowpass(out)))
        
        out_real = self.pool2(self.relu2(self.conv1_real(out)))
        out_imag = self.pool3(self.relu3(self.conv1_imag(out)))
        mag = torch.sqrt(out_real**2+out_imag**2+self.eps)
        
        result = torch.cat([out_lowpass, mag], axis=1)
        return result.view(-1, 9*C, result.shape[2], result.shape[3])        

# remove MaxPool layer
class ModelLevel1_Learn_I1(nn.Module):
    def __init__(self):
        super(ModelLevel1_Learn_I1, self).__init__()
        self.conv1_lowpass = nn.Conv2d(1,1,5,padding=2, stride=2)
        
        self.conv1_real = nn.Conv2d(1,8,5,padding=2, stride=2)
        self.conv1_imag = nn.Conv2d(1,8,5,padding=2, stride=2)
        self.eps = 1e-4
        
    def forward(self, x):
        out = x.view(-1, 1, x.shape[2], x.shape[3])
        out_lowpass = self.conv1_lowpass(out)        
        mag = torch.sqrt(self.conv1_real(out)**2 + self.conv1_imag(out)**2 + self.eps)
        result = torch.cat([out_lowpass, mag], axis=1)
        return result.view(-1, 9*C, result.shape[2], result.shape[3])    
    
# the equivalent output structure with level 2 scattering transform
class ModelLevel2(nn.Module):
    def __init__(self):
       super(ModelLevel2, self).__init__()
       self.conv1 = nn.Conv2d(1,9,5,padding=2)
       self.relu1 = nn.ReLU()
       self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
       
       self.conv2 = nn.Conv2d(1,9,5,padding=2)
       self.relu2 = nn.ReLU()
       self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
       
    def forward(self, x):
        x = x.view(-1, 1, x.shape[2], x.shape[3])
        out = self.pool1(self.relu1(self.conv1(x)))
        
        out = out.view(-1, 1, out.shape[2], out.shape[3])
        out = self.pool2(self.relu2(self.conv2(out)))
        
        out = out.view(-1, 81*C, out.shape[2], out.shape[3])
        return out

# add BatchNorm2d
class ModelLevel2_I0(nn.Module):
    def __init__(self):
        super(ModelLevel2_I0, self).__init__()
        self.conv1 = nn.Conv2d(1,9,5,padding=2)
        self.bn1 = nn.BatchNorm2d(9)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
           
        self.conv2 = nn.Conv2d(1,9,5,padding=2)
        self.bn2 = nn.BatchNorm2d(9)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        
    def forward(self, x):
        x = x.view(-1, 1, x.shape[2], x.shape[3])
        out = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        
        out = out.view(-1, 1, out.shape[2], out.shape[3])
        out = self.pool2(self.relu2(self.bn2(self.conv2(out))))
        
        out = out.view(-1, 81*C, out.shape[2], out.shape[3])
        return out

# Remove MaxPool layer
class ModelLeve2_I1(nn.Module):
    def __init__(self):
        super(ModelLeve2_I1, self).__init__()
        self.conv1 = nn.Conv2d(1,9,5,padding=2, stride=2)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(1,9,5,padding=2, stride=2)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = x.view(-1, 1, x.shape[2], x.shape[3])
        out = self.relu1(self.conv1(out))
        
        out = out.view(-1,1, out.shape[2], out.shape[3])
        out = self.relu2(self.conv2(out))
        
        out = out.view(-1, 81*C, out.shape[2], out.shape[3])
        return out

# Remove MaxPool & ReLU layers    
class ModelLeve2_I2(nn.Module):
    def __init__(self):
        super(ModelLeve2_I2, self).__init__()
        self.conv1 = nn.Conv2d(1,9,5,padding=2, stride=2)
        self.conv2 = nn.Conv2d(1,9,5,padding=2, stride=2)
        
    def forward(self, x):
        out = x.view(-1, 1, x.shape[2], x.shape[3])
        out = self.conv1(out)
        
        out = out.view(-1,1, out.shape[2], out.shape[3])
        out = self.conv2(out)
        
        out = out.view(-1, 81*C, out.shape[2], out.shape[3])
        return out    

# Add 2 components: real and imag part.    
class ModelLevel2_I3(nn.Module):
    def __init__(self):
        super(ModelLevel2_I3, self).__init__()
        self.conv1_lowpass = nn.Conv2d(1,1,5,padding=2,stride=2)
        
        self.conv1_real = nn.Conv2d(1,8,5,padding=2,stride=2)
        self.conv1_imag = nn.Conv2d(1,8,5,padding=2,stride=2)
        self.eps1 = 1e-4
        
        self.conv2_lowpass = nn.Conv2d(1,1,5,padding=2,stride=2)
        
        self.conv2_real = nn.Conv2d(1,8,5,padding=2,stride=2)
        self.conv2_imag = nn.Conv2d(1,8,5,padding=2,stride=2)
        self.eps2 = 1e-4        
        
    def forward(self, x):
        out = x.view(-1, 1, x.shape[2], x.shape[3])
        
        out1_lowpass = self.conv1_lowpass(out)
        out1_mag = torch.sqrt(self.conv1_real(out)**2 + self.conv1_imag(out)**2 + self.eps1)
        out = torch.cat([out1_lowpass, out1_mag], axis=1)
        
        out = out.view(-1, 1, out.shape[2], out.shape[3])
        out2_lowpass = self.conv2_lowpass(out)
        out2_mag = torch.sqrt(self.conv2_real(out)**2 + self.conv2_imag(out)**2 + self.eps2)
        out = torch.cat([out2_lowpass, out2_mag], axis=1)
        
        return out.view(-1, 81*C, out.shape[2], out.shape[3])        


from collections import OrderedDict
# default for color image
class ModelLevel2_I4(nn.Module):
    def __init__(self):
        super(ModelLevel2_I4, self).__init__()
        self.extract = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(3, 27, kernel_size=5, padding=2)),
                    ('relu1', nn.ReLU()),
                    ('pool1', nn.MaxPool2d(kernel_size=(2, 2))),                    
                    
                    ('conv2', nn.Conv2d(27, 81, kernel_size=5, padding=2)),
                    ('relu2', nn.ReLU()),
                    ('pool2', nn.MaxPool2d(kernel_size=(2, 2))),                    
                    
                    ('conv3', nn.Conv2d(81, 243, kernel_size=5, padding=2)),
                    ('relu3', nn.ReLU())
                ]))            
        
    def forward(self, x):
        return self.extract(x)
        
# =============================================================================
# #model = ModelLevel1()        
# #model = ModelLevel1_I1() 
# #model = ModelLevel1_Learn()
# model = ModelLevel1_Learn_I1()
#         
# #model = ModelLevel2()
# #model = ModelLevel2_I1()
# #model = ModelLevel2_I2()
# model = ModelLevel2_I3()
# 
# x = torch.randn(2,C,512,128)
# y = model(x)
# print(y.shape)
# =============================================================================
