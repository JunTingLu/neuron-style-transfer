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
generated_image=original_image.clone().requires_grad_(True)
optimizer=optim.Adam([generated_image],lr=0.004)
epoch=400

for e in range (epoch):
    gen_features=model(generated_image) 
    orig_features=model(original_image)
    style_features=model(style_image) 

    total_loss=calculate_loss(gen_features, orig_features, style_features)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step() # 每個epoch更新generated_image 參數

    if e==epoch-1:
        save_image(generated_image,"resnet_gen_{}.png".format(e))    