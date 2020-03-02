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

from our_model_utils_paper import AttrDict, show, generate_image, train, test, evaluate_model, display_classified_images
import our_model_utils_paper as MU

MU.class_names = ['cr', 'gg', 'in', 'pa', 'ps', 'rp', 'rs', 'sc', 'sp']
#kwargs = {'num_workers': 0, 'pin_memory': False} if MU.use_cuda else {}

MU.imgsize = (64,64)
MU.args.batch_size      = 3072
MU.args.test_batch_size = 3072
MU.args.epochs = 1

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


from our_model import ModelLevel2, ModelLevel2_I3, C
myLayer = ModelLevel2()
x = torch.randn(MU.args.batch_size, C, MU.imgsize[0], MU.imgsize[1])
y = myLayer(x)
print(y.shape)
feature_num = y.size(1) * y.size(2) * y.size(3)
print(feature_num)


from collections import OrderedDict
class OurClassifier(nn.Module):
    def __init__(self):
        super(OurClassifier, self).__init__()
        # Extract features
        self.conv1 = nn.Conv2d(1,9,5,padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
       
        self.conv2 = nn.Conv2d(1,9,5,padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        # Classifer
        self.classifer = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(feature_num, 512)),
                ('reluC1', nn.ReLU()),
                ('linear2', nn.Linear(512, 9)),
                ('reluC2', nn.ReLU())
                ]))
        
    def forward(self, x):
        out = x.view(-1, 1, x.shape[2], x.shape[3])
        out = self.pool1(self.relu1(self.conv1(out)))
        
        out = out.view(-1, 1, out.shape[2], out.shape[3])
        out = self.pool2(self.relu2(self.conv2(out)))
        out = out.view(-1, 81*C, out.shape[2], out.shape[3])
        
        out = out.view(out.size(0), -1)
        out = self.classifer(out)
        return F.log_softmax(out)

from our_classifer import OurLinear_I1, Classifer_paper

class OurClassifier_I1(nn.Module):
    def __init__(self):
        super(OurClassifier_I1, self).__init__()
        
        # extract features
        self.ourExtractFeatures = ModelLevel2()
        #self.our = ModelLevel2_I3()

# =============================================================================
#         # Classifer one linear layer
#         self.classifer = OurLinear_I1(feature_num, 9)
# =============================================================================
        
        # Classifer multi linear layers
        self.classifer = Classifer_paper(feature_num, 9)
        
    def forward(self, x):
        out = self.ourExtractFeatures(x)
        
        out = out.view(out.size(0), -1)
        out = self.classifer(out)
        return F.log_softmax(out)    
        
# =============================================================================
# #clsModel = OurClassifier()
# clsModel = OurClassifier_I1()    
# y_pred = clsModel(x)
# print(clsModel.our.conv1.weight.shape)
# =============================================================================
        

# Train & test
#classifier = OurClassifier()
classifier = OurClassifier_I1()
if MU.use_cuda : classifier.cuda()
evaluate_model(classifier)

import datetime

PATH = 'model/layers_{}.pth'.format(str(datetime.datetime.now()))
torch.save(classifier.state_dict(), PATH)

