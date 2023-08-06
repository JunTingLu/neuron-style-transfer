import torch
import matplotlib.pyplot as plt

#initialize the parameters required for fitting the model
alpha=1  # content weight
beta=100 #style weight


def calc_content_loss(gen_feat,orig_feat):
    #calculating the content loss 
    content_l=torch.mean((gen_feat-orig_feat)**2) 
    return content_l

def calc_style_loss(gen,style):
    #Calculating the gram matrix for the style and the generated image
    batch,channel,height,width=gen.shape
    G=torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
    A=torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())
    # Calculating the style loss (batch-normalization)
    style_l=torch.mean((G-A)**2) #/(4*channel*(height*width)**2)
    # print(21,G.shape)

    # instance normalization
    # Reshape the tensors to (batch, channel, height * width)
    # gen_reshaped = gen.view(batch, channel, height * width)
    # style_reshaped = style.view(batch, channel, height * width)
    
    # # Calculate the mean and variance for each sample,we will get (batch, channel, 1)shape
    # mean_G = torch.mean(gen_reshaped, dim=2, keepdim=True)
    # mean_A = torch.mean(style_reshaped, dim=2, keepdim=True)
    
    # var_G = torch.var(gen_reshaped, dim=2, keepdim=True)
    # var_A = torch.var(style_reshaped, dim=2, keepdim=True)
    
    # # Instance normalization for each sample
    # G_normalized = (gen_reshaped - mean_G) / torch.sqrt(var_G + 1e-5)
    # A_normalized = (style_reshaped - mean_A) / torch.sqrt(var_A + 1e-5)
    
    # # Calculate the style loss (instance normalization)
    # style_l = torch.mean((G_normalized - A_normalized) ** 2) / (height * width)
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


