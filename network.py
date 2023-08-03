from collections import namedtuple
import torchvision.models as models
from torch import nn
import torch

#Assigning the GPU to the variable device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vggoutput = namedtuple("vggoutput", ["relu1_2", "relu2_2", "relu3_2", "relu4_2"])
model=models.vgg19(pretrained=True)

# VGGnet
class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()

        # 取vgg前29層的卷積層輸出
        # self.req_features= ['0','5','10','19','28'] 
        self.layer_names= ['3','8','13','20'] 
        #Since we need only the 5 layers in the model so we will be dropping all the rest layers from the features of the model
        self.model=models.vgg19(pretrained=True).features[:29] #model will contain the first 29 layers       
 
    # x holds the input tensor(image) that will be feeded to each layer
    def forward(self,x):
        features=[]
        # features={}
        for layer_num,layer in enumerate(self.model):
            #activation of the layer will stored in x
            x=layer(x)
            #appending the activation of the selected layers and return the feature array
            if (str(layer_num) in self.layer_names):
                features.append(x)    
            # if  layer_num in self.layer_name:
            #     features.append(x)
        return features
        # return vggoutput(**features)
    


# Normalization
# create a module to normalize input image so we can easily put it in a nn.Sequential
# class normalization(nn.Module):
# 	def __init__(self):
# 		super(normalization, self).__init__()
# 		# .view the mean and std to make them [C x 1 x 1] so that they can
# 		# directly work with image Tensor of shape [B x C x H x W].
# 		# B is batch size. C is number of channels. H is height and W is width.
# 		mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
# 		std = torch.tensor([0.229, 0.224, 0.225]).to(device)
# 		self.mean = torch.tensor(mean).view(-1, 1, 1)
# 		self.std = torch.tensor(std).view(-1, 1, 1)

# 	def forward(self, img):
# 		# normalize img
# 		return (img - self.mean) / self.std


# Resbet
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        # indicate 6 layers for test
        self.req_features = ['0' for _ in range(6)]
        # Load the pre-trained ResNet50 model
        resnet_model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(resnet_model.children())[:6])

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            #import ipdb; ipdb.set_trace()
            # Activation of the layer will be stored in x
            x = layer(x)
            if str(layer_num) in self.req_features:
                features.append(x)
        return features


#Load the model to the GPU ( eval() --> vgg function 參數不會更動)  
# model = nn.Sequential(normalization)  
model=VGG().to(device).eval()    
# model=ResidualBlock().to(device).eval() 