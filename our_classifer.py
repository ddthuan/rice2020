# https://towardsdatascience.com/how-to-build-your-own-pytorch-neural-network-layer-from-scratch-842144d623f6

import torch
from torch import nn
from collections import OrderedDict

class OurLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))       
        
    def forward(self, input):
        x, y = input.shape
# =============================================================================
#         if y != self.in_features:
#             sys.exit(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
# =============================================================================
        output = input @ self.weight.t() + self.bias
        return output    

import math

class OurLinear_I1(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, input):
        x, y = input.shape
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        output = input.matmul(self.weight.t())
        if self.bias is not None:
            output += self.bias
        ret = output
        return ret
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )   
        
class Classifer_paper(nn.Module):
    def __init__(self, feature_num, class_num):
        super(Classifer_paper, self).__init__()
        self.feature_num = feature_num
        self.class_num = class_num

        self.classifer = nn.Sequential(OrderedDict([
                ('linear1', OurLinear_I1(self.feature_num, 1024)),
                ('linear2', OurLinear_I1(1024, 512)),
                ('linear3', OurLinear_I1(512, self.class_num))
                ]))        
    
    def forward(self, x):
        return self.classifer(x)    
    
# =============================================================================
# feature_num = 5000
# class_num = 128
# model = Classifer_paper(feature_num, class_num)
# #model = OurLinear(feature_num, class_num)
# #model = OurLinear_I1(feature_num, class_num)
# 
# x = torch.randn(10,feature_num)
# y = model(x)
# print(y.shape)
# =============================================================================

    