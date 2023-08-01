import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import time
import wandb 
from PIL import Image


run = wandb.init(
  project= "NST"
)

start_time = time.time()
#Loadung the model vgg19 that will serve as the base model
#model=models.vgg19(pretrained=True).features

#Assigning the GPU to the variable device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defing a function that will load the image and perform the required preprocessing and put it on the GPU
def image_loader(path,is_cuda=False):
    image=Image.open(path)
    #defining the image transformation steps to be performed before feeding them to the model
    loader=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])
    #The preprocessing steps involves resizing the image and then converting it to a tensor
    image=loader(image).unsqueeze(0)
    return image.to(device,torch.float)


# Network 
# Resblock
# class ResidualBlock(nn.Module):
#     def __init__(self, in_features, expansion=1, num_blocks=2,use_dropout=False):
#         super(ResidualBlock, self).__init__()
#         layers = []
#         for _ in range(num_blocks):
#             layers += [
#                 nn.ReflectionPad2d(1),
#                 nn.Conv2d(in_features * expansion, in_features, 3),
#                 nn.InstanceNorm2d(in_features),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.ReflectionPad2d(1),
#                 nn.Conv2d(in_features * expansion, in_features, 3),
#                 nn.InstanceNorm2d(in_features),
#             ]
#             # to avoid over-fitting (only left 80%)
#             if use_dropout:
#                 layers += [nn.Dropout(0.2)]
#         """ use Sequential collect "list" of layers"""
#         self.block = nn.Sequential(*layers)

#     def forward(self, x):
#         return x + self.block(x)


# Resnet
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


# VGGnet
class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        # 取vgg前29層的卷積層輸出
        self.req_features= ['0','5','10','19','28'] 
        #Since we need only the 5 layers in the model so we will be dropping all the rest layers from the features of the model
        self.model=models.vgg19(pretrained=True).features[:29] #model will contain the first 29 layers       
    # x holds the input tensor(image) that will be feeded to each layer
    def forward(self,x):
        features=[]
        for layer_num,layer in enumerate(self.model):
            #activation of the layer will stored in x
            x=layer(x)
            #appending the activation of the selected layers and return the feature array
            if (str(layer_num) in self.req_features):
                features.append(x)    
        return features


#Loading the original and the style image
original_image=image_loader("input/content/Sakura.jpg")
style_image=image_loader("input/style/vango.jpg")

#Creating the generated image from the original image (copy 原圖)
generated_image=original_image.clone().requires_grad_(True)


# 原圖片加入random noise (initialization, for ex:white or gaussian)
def gaussian_noise(image_tensor,mean=0,std=1):
    noise=torch.randn_like(image_tensor)*std+mean
    # clamp返回新的張量，且限制張量中的min/max在0-255
    noisy_img=image_tensor+noise
    return noisy_img


def calc_content_loss(gen_feat,orig_feat):
    #calculating the content loss of each layer by calculating the MSE between the content and generated features and adding it to content loss
    content_l=torch.mean((gen_feat-orig_feat)**2)#*0.5
    return content_l

def calc_style_loss(gen,style):
    #Calculating the gram matrix for the style and the generated image
    batch,channel,height,width=gen.shape
    G=torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
    A=torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())
        
    #Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
    style_l=torch.mean((G-A)**2)#/(4*channel*(height*width)**2)
    return style_l

def calculate_loss(gen_features, orig_feautes, style_featues):
    style_loss=content_loss=0
    for gen,cont,style in zip(gen_features,orig_feautes,style_featues):
        #extracting the dimensions from the generated image
        content_loss+=calc_content_loss(gen,cont)
        style_loss+=calc_style_loss(gen,style)
    
    #calculating the total loss of e th epoch
    total_loss=alpha*content_loss + beta*style_loss 
    return total_loss


#Load the model to the GPU ( eval() --> vgg function 參數不會更動)
# model=VGG().to(device).eval() 
model=ResidualBlock().to(device).eval() 

#initialize the parameters required for fitting the model
epoch=1000
lr=0.004
# ratio of apha/beta (1e1, 1e2, 1e-1, 1e-2),
const=1e2
alpha=1 # content weight
beta=const/alpha #style weight

#using adam optimizer and it will update the generated image not the model parameter 
optimizer=optim.Adam([generated_image],lr=lr)

#iterating for 1000 times
for e in range (epoch):
    #extracting the features of generated, content and the original required for calculating the loss
    gen_features=model(generated_image) 
    orig_feautes=model(original_image)
    style_featues=model(style_image) 
    
    #iterating over the activation of each layer and calculate the loss and add it to the content and the style loss
    total_loss=calculate_loss(gen_features, orig_feautes, style_featues)
    #optimize the pixel values of the generated image and back-propagate the loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step() # 每個epoch更新generated_image 參數
    wandb.log({"Loss": total_loss})

    if e==900:
        print(177,total_loss)
        save_image(generated_image,"resnet_gen.png")

end_time = time.time()
execution_time = end_time - start_time
print(f"Time taken: {execution_time:.5f} seconds")

with open("CNN_model.pth", "wb") as f:
    torch.save(model.state_dict(), f)


# show images with 5*2 
def show_original_and_gray_images(image_path):
    try:
        # 加载原始图像
        original_image = Image.open(image_path)
        # 将图像转换为灰度图像
        gray_image = original_image.convert("L")
        # 调整大小为5*2
        original_image = original_image.resize((5, 2))
        gray_image = gray_image.resize((5, 2))
        # 显示原始图像
        print("Original Image:")
        original_image.show()
        # 显示灰度图像
        print("\nGray Image:")
        gray_image.show()

    except Exception as e:
        print("Error: ", str(e))

# 调用函数并传入图像路径
# image_path = "path/to/your/image.jpg"
# show_original_and_gray_images(image_path)


#%%
# 啟動tensorboard查看訓練曲線
""" inspect log """
# terminal 輸入
# %wandb login
# https://docs.wandb.ai/juntinglu
# get your API
        

