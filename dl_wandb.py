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

# training records
run = wandb.init(
  project= "NST"
)

start_time = time.time()

# Image resizing
def image_loader(path):
    image=Image.open(path)
    loader=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])
    #Converting it to a tensor
    image=loader(image).unsqueeze(0)
    return image.to(device,torch.float)


#Loading the original and the style image
original_image=image_loader("input/content_3/me.jpg")
style_image=image_loader("input/style_3/vango.jpg")

#Creating the generated image from the original image (copy from original image)
generated_image=original_image.clone().requires_grad_(True)


# random noise (initialization, for ex:white or gaussian)
# def gaussian_noise(image_tensor,mean=0,std=1):
#     noise=torch.randn_like(image_tensor)*std+mean
#     noisy_img=image_tensor+noise
#     return noisy_img


#Update the generated image not the model parameter 
lr=0.004
optimizer=optim.Adam([generated_image],lr=lr)
epoch=1000

# training 
for e in range (epoch):
    # loss calculation
    gen_features=model(generated_image) 
    orig_features=model(original_image)
    style_features=model(style_image) 
    total_loss=calculate_loss(gen_features, orig_features, style_features)
    #optimize the generated image and back-propagate the loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step() # 每個epoch更新generated_image 參數
    wandb.log({"Loss": total_loss})

    if e==epoch-1:
        save_image(generated_image,"resnet_gen_{}.png".format(e))

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
        

