from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
import itertools

# Performance monitoring
from time import process_time
import matplotlib.pyplot as plt
import numpy as np

# Disable warnings from the Scattering transform...
import warnings
warnings.filterwarnings("ignore")

# Train and visualize the performances of our models

#from model_utils import AttrDict, show, generate_image, train, test, evaluate_model, display_classified_images
#import model_utils_wavelet as MU

from method_model_utils import AttrDict, show, generate_image, train, test, evaluate_model, display_classified_images
import method_model_utils as MU

#MU.display_parameters()

MU.class_names = ['cr', 'gg', 'in', 'pa', 'ps', 'rp', 'rs', 'sc', 'sp']
#kwargs = {'num_workers': 0, 'pin_memory': False} if MU.use_cuda else {}

MU.imgsize = (64,64)
MU.args.batch_size      = 128
MU.args.test_batch_size = 128
MU.args.epochs = 150

num_classes = 9
data_dir = "steel/"

data_transforms = {
    'train': transforms.Compose(
            
            [#transforms.Resize([28,28],2),
                                 transforms.ToTensor(),                                 
                                 #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ]),
    'val': transforms.Compose(            
            [
                    transforms.ToTensor(),
                                 #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])
}

validation_split = .20

# Chia tap train va test doc lap
dataset = torchvision.datasets.ImageFolder(data_dir, data_transforms['train'])
train_size = int((1-validation_split) * len(dataset))
test_size = len(dataset) - train_size
MU.train_dataset, MU.test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Iterator for the training pass:
MU.train_loader = torch.utils.data.DataLoader( MU.train_dataset,
    batch_size=MU.args.batch_size, shuffle=True, # Data is drawn as randomized minibatches
    )                                 # Practical settings for parallel loading

# Iterator for the testing pass, with the same settings:
MU.test_loader  = torch.utils.data.DataLoader( MU.test_dataset,
    batch_size=MU.args.test_batch_size, shuffle=True, 
    )


from collections import OrderedDict
device = torch.device("cuda:0")

class Scattering2dNet(nn.Module):
    def __init__(self):
        super(Scattering2dNet, self).__init__()

        self.convNext = nn.Sequential(OrderedDict([
                    ('conv4', nn.Conv2d(243, 243, kernel_size=5, padding=2)),
                    #('bn1', nn.BatchNorm2d(243)),
                    ('pool4', nn.MaxPool2d(kernel_size=(2, 2))),
                    ('relu4', nn.ReLU())
                ]))
        
        #1296 = 243 * 4 * 4
        self.fc = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(15552, 1024)),
            ('relu5', nn.ReLU()),
            ('linear2', nn.Linear(1024, 512)),
            ('relu6', nn.ReLU()),
            ('linear3', nn.Linear(512, 256)),
            ('relu7', nn.ReLU()),
            ('linear4', nn.Linear(256, 9)),
            ('relu8', nn.ReLU()),
            #('sig7', nn.LogSoftmax(dim=-1))
        ]))
    
    def forward(self, x):
        #print(x.shape)    
        #output = self.bn(x.view(-1, 512, 8, 8).to(device))
        output = x.view(-1, 243, 16, 16).to(device)
        #print(output.shape)
        
        output = self.convNext(output)
    
        output = output.view(output.size(0), -1)
        
        output = self.fc(output)
        #return output
        return F.log_softmax(output) 


# Giam so luong layer
class Scattering2dNet_Good(nn.Module):
    def __init__(self):
        super(Scattering2dNet_Good, self).__init__()        

        self.fc = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(62208, 512)),
            ('relu5', nn.ReLU()),
            ('linear4', nn.Linear(512, 9)),
            ('relu8', nn.ReLU()),
        ]))
    
    def forward(self, x):
        #print(x.shape)    
        #output = self.bn(x.view(-1, 512, 8, 8).to(device))
        output = x.view(-1, 243, 16, 16).to(device)
        #print(output.shape) [128,243,16,16]
        
        #output = self.convNext(output)
    
        output = output.view(output.size(0), -1)
        
        output = self.fc(output)
        #return output
        return F.log_softmax(output) 
    

# Giam so luong layer
class Scattering2dNet_Good_Cuda(nn.Module):
    def __init__(self):
        super(Scattering2dNet_Good_Cuda, self).__init__()        
        
        #1296 = 243 * 4 * 4
        self.classifer = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(62208, 1024)),
            ('relu5', nn.ReLU()),
# =============================================================================
#             ('linear2', nn.Linear(10240, 1024)),
#             ('relu6', nn.ReLU()),
# =============================================================================
            ('linear3', nn.Linear(1024, 512)),
            ('relu7', nn.ReLU()),
            ('linear4', nn.Linear(512, 256)),
            ('relu8', nn.ReLU()),
            ('linear5', nn.Linear(256, 9)),
            ('relu9', nn.ReLU()),
            #('sig7', nn.LogSoftmax(dim=-1))
        ]))
    
    def forward(self, x):
        #print(x.shape)    
        #output = self.bn(x.view(-1, 512, 8, 8).to(device))
        output = x.view(-1, 243, 16, 16).to(device)
        #print(output.shape) [128,243,16,16]
        
        #output = self.convNext(output)
    
        output = output.view(output.size(0), -1)
        
        output = self.classifer(output)
        #return output
        return F.log_softmax(output) 



# Giam so luong layer
class Scattering2dNet_Good_Cuda_hiden(nn.Module):
    def __init__(self):
        super(Scattering2dNet_Good_Cuda_hiden, self).__init__()        
        
        self.enhance = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(243, 243, kernel_size=3, padding=1)),
            #('pool3', nn.MaxPool2d(kernel_size=(2, 2))),
            #('dropout1', nn.Dropout2d()),
            ('relu3', nn.ReLU()),
            
            ('conv4', nn.Conv2d(243, 243, kernel_size=3, padding=1)),
            #('pool4', nn.MaxPool2d(kernel_size=(2, 2))),
            #('dropout2', nn.Dropout2d()),
            ('relu4', nn.ReLU()),            
        ]))      
    
        #1296 = 243 * 4 * 4
        self.classifer = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(62208, 1024)),
            #('linear1', nn.Linear(31104, 1024)),
            #('linear1', nn.Linear(15552, 1024)),
            ('relu5', nn.ReLU()),
# =============================================================================
#             ('linear2', nn.Linear(10240, 1024)),
#             ('relu6', nn.ReLU()),
# =============================================================================
            ('linear3', nn.Linear(1024, 512)),
            ('relu7', nn.ReLU()),
            ('linear4', nn.Linear(512, 256)),
            ('relu8', nn.ReLU()),
            ('linear5', nn.Linear(256, 9)),
            ('relu9', nn.ReLU()),
            #('sig7', nn.LogSoftmax(dim=-1))
        ]))
    
        torch.nn.init.xavier_uniform_(self.enhance[0].weight)
        torch.nn.init.xavier_uniform_(self.enhance[2].weight)
        
    
    def forward(self, x):
        #print(x.shape)    
        #output = self.bn(x.view(-1, 512, 8, 8).to(device))
        output = x.view(-1, 243, 16, 16).to(device)
        #print(output.shape) [128,243,16,16]
        
        output = self.enhance(output)
    
        output = output.view(output.size(0), -1)
        
        output = self.classifer(output)
        #return output
        return F.log_softmax(output) 


from our_classifer import Classifer_paper, OurLinear_I1
# Giam so luong layer
class Scattering2dNet_Cuda_I1(nn.Module):
    def __init__(self):
        super(Scattering2dNet_Cuda_I1, self).__init__()        
        
        #1296 = 243 * 16 * 16
        self.classifer = OurLinear_I1(62208,9)
        
        # 3 linear layer for classifier
        #self.classifer = Classifer_paper(62208,9)
    
    def forward(self, x):
        #print(x.shape)    
        #output = self.bn(x.view(-1, 512, 8, 8).to(device))
        output = x.view(-1, 243, 16, 16).to(device)
        #print(output.shape) [128,243,16,16]
        
        #output = self.convNext(output)    
        output = output.view(output.size(0), -1)
        
        output = self.classifer(output)
        #return output
        return F.log_softmax(output)     

classifier = Scattering2dNet_Cuda_I1()
if MU.use_cuda : classifier.cuda()
evaluate_model(classifier)

import datetime
PATH = 'model/scatter_{}.pth'.format(str(datetime.datetime.now()))
torch.save(classifier.state_dict(), PATH)



