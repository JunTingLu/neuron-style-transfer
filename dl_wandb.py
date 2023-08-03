import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.optim as optim
from torchvision.utils import save_image
import time
import wandb 
from PIL import Image
from utils import *
from network import *

run = wandb.init(
  project= "NST"
)

start_time = time.time()


# defing a function that will load the image and perform the required preprocessing and put it on the GPU
def image_loader(path):
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


#Loading the original and the style image
original_image=image_loader("input/content/Sakura.jpg")
style_image=image_loader("input/style/vango.jpg")

#Creating the generated image from the original image (copy 原圖)
generated_image=original_image.clone().requires_grad_(True)


# 原圖片加入random noise (initialization, for ex:white or gaussian)
def gaussian_noise(image_tensor,mean=0,std=1):
    noise=torch.randn_like(image_tensor)*std+mean
    noisy_img=image_tensor+noise
    return noisy_img


#using adam optimizer and it will update the generated image not the model parameter 
lr=0.004
optimizer=optim.Adam([generated_image],lr=lr)
epoch=1000

#iterating for 1000 times
for e in range (epoch):
    #extracting the features of generated, content and the original required for calculating the loss
    gen_features=model(generated_image) 
    orig_features=model(original_image)
    style_features=model(style_image) 
    
    #iterating over the activation of each layer and calculate the loss and add it to the content and the style loss
    total_loss=calculate_loss(gen_features, orig_features, style_features)
    #optimize the pixel values of the generated image and back-propagate the loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step() # 每個epoch更新generated_image 參數
    wandb.log({"Loss": total_loss})

    if e==epoch-1:
        print(177,total_loss)
        save_image(generated_image,"resnet_gen.png")

end_time = time.time()
execution_time = end_time - start_time
print(f"Time taken: {execution_time:.5f} seconds")

with open("CNN_model.pth", "wb") as f:
    torch.save(model.state_dict(), f)



#%%
# 啟動tensorboard查看訓練曲線
""" inspect log """
# terminal 輸入
# %wandb login
# https://docs.wandb.ai/juntinglu
# get your API
        

