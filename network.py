from collections import namedtuple
import torchvision.models as models
from torch import nn
import torch

#Assigning the GPU to the variable device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=models.vgg19(pretrained=True)

# vgg layers
# layers = (
#         'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
#         'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
#         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
#         'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
#         'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
#         'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
#         'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
#         'relu5_3', 'conv5_4', 'relu5_4'
#     )

# VGGnet
class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__() 
            # '3':  "relu1_2",
            # '8':  "relu2_2",
            # '13': "relu3_2",
            # '20': "relu4_2"
            # '29': "relu5_2"
        self.layer_names= ['1','3','8','13','20','29'] 
        self.model=models.vgg19(pretrained=True).features[:30] #model will contain the first 30 layers       
 
    def forward(self,x):
        features=[]
        # features={}
        for layer_num,layer in enumerate(self.model):
            #activation of the layer will stored in x
            x=layer(x)
            #appending the activation of the selected layers and return the feature array
            if (str(layer_num) in self.layer_names):
                features.append(x)    
        return features
    

# Resnet
# class ResidualBlock(nn.Module):
#     def __init__(self):
#         super(ResidualBlock, self).__init__()
#         # indicate 6 layers for test
#         self.req_features = ['0' for _ in range(6)]
#         # Load the pre-trained ResNet50 model
#         resnet_model = models.resnet50(pretrained=True)
#         self.model = nn.Sequential(*list(resnet_model.children())[:6])

#     def forward(self, x):
#         features = []
#         for layer_num, layer in enumerate(self.model):
#             #import ipdb; ipdb.set_trace()
#             # Activation of the layer will be stored in x
#             x = layer(x)
#             if str(layer_num) in self.req_features:
#                 features.append(x)
#         return features



#Load the model to the GPU ( eval() --> vgg function 參數不會更動)  
model=VGG().to(device).eval()    
# model=ResidualBlock().to(device).eval() 