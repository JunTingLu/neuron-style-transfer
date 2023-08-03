import torch
import matplotlib.pyplot as plt

#initialize the parameters required for fitting the model
alpha=1  # content weight
beta=1000 #style weight


def calc_content_loss(gen_feat,orig_feat):
    #calculating the content loss of each layer by calculating the MSE between the content and generated features and adding it to content loss
    content_l=torch.mean((gen_feat-orig_feat)**2) #*0.5
    return content_l

def calc_style_loss(gen,style):
    #Calculating the gram matrix for the style and the generated image
    batch,channel,height,width=gen.shape
    G=torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
    A=torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())
    #Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
    style_l=torch.mean((G-A)**2) #/(4*channel*(height*width)**2)
    # style_l=((G-A)**2)/(channel*height*width)
    return style_l


def calculate_loss(gen_features, orig_features, style_features):
    style_loss=content_loss=0
    for gen,cont,style in zip(gen_features,orig_features,style_features):
        #extracting the dimensions from the generated image
        content_loss+=calc_content_loss(gen,cont)
        style_loss+=calc_style_loss(gen,style)
    
    #calculating the total loss of e th epoch
    total_loss=alpha*content_loss + beta*style_loss 
    return total_loss



# show images
def show_original_and_gray_images(img):
    plt.imshow(img)
    plt.show()
