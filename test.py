import torch
from network import model
from dl_wandb import image_loader,calculate_loss
from torchvision.utils import save_image
import torch.optim as optim
from utils import *

with open("CNN_model.pth", "rb") as f:
    torch.load(model.state_dict(), f)

#Loading the original and the style image
original_image=image_loader("input/content/Sakura.jpg")
style_image=image_loader("input/style/vango.jpg")

#Creating the generated image from the original image (copy 原圖)
generated_image=original_image.clone().requires_grad_(True)

#using adam optimizer and it will update the generated image not the model parameter 
optimizer=optim.Adam([generated_image],lr=0.004)
epoch=400

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

    if e==epoch-1:
        save_image(generated_image,"resnet_gen_{}.png".format(e))    