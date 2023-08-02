# Image Style Transfer
圖像轉換(Style transfer)一直是個讓人感到新穎的主題，本文利用CNN(Convolutional Neural Networks)的方式進行圖片的風格轉換，並藉由調整參數來決定原圖像與轉換風格後的相似程度，細節將在本文陸續說明，這裡分享自己實作上的過程與結果

## 簡介
> 圖像轉換(Style transfer)最早可追溯到2015年Gatys 等人所發表的 A Neural Algorithm of Artistic Style，
> 他們所採用的方式是利用VGG(Visual Geometry Group)模型進行圖像的特徵提取，關鍵在於提取出來的特徵分為content 和 style features，所謂 content是指一張圖像的大致輪廓，而style是指圖像中更細節的資訊，因此只要將原圖像的content成分取出，搭配欲產生的風格照片之style進行結合，透過loss函數的設計在這兩著間達成平衡，便能合成出具有content和style成分的圖象。

## 技術與原理
> 整個模型的主要核心在於如何體取出圖像中的content和style的特徵，接著透過增加圖像預處理(image preprocessing)，以及嘗試不同的模型架構、learning rate的選擇、調整loss參數達到最佳的合成效果

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*p15iAAgqiCyVAbi4msgfeQ.png)
(引用自參考資料[3])

- ## **圖像預處理**
> 這裡嘗試先將圖像進行縮放和歸一化，使用pytorch中的transform套件進行縮放，並轉為tensor的形式，接著在image 的部分即是將原圖像套用到transform定義好的縮放方式，並從原來的(512,512,3)在第0維上新增一個維度，形成(1,512,512,3)的四維向量，目的是為了方便後續進行特徵(features)的堆疊。另外這裡的(512,512,3)分別代表圖像的512x512的像素及RGB三顏色(通道數)。
> 加入高斯噪聲後






```
def image_loader(path,is_cuda=False):
    image=Image.open(path)
    loader=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])
    image=loader(image).unsqueeze(0)
    return image.to(device,torch.float)
```

- ## **模型架構**
> 首先從模型架構來說，我採用VGG19的預訓練(pre-training)模型，直接利用前一大段的CNN架構來加快模型收斂時間。這裡只保留VGG前29層，其中會使用到的五個的filter，去輸出的圖像並依序取出features並存在feature box中，直觀上可以想像為了讓機器學會辨識一張圖像的特徵(例如:紋理、邊緣等等資訊)，在VGG模型中透過不同層濾波器(filter)所產生的不同特徵圖，又稱為feature map，而feature box就是收集這些feature map的過程。如下示意圖

![](https://upscfever.com/upsc-fever/en/data/deeplearning4/images/NST_LOSS.png)
(引用自參考資料[4])

```
class VGG(nn.Module):
def __init__(self,is_cuda):
    self.is_cuda=is_cuda
    super(VGG,self).__init__()
    self.req_features= ['0','5','10','19','28'] 
    self.model=models.vgg19(pretrained=True).features[:29] 

def forward(self,x):
    features=[]
    #Iterate over all the layers of the mode
    for layer_num,layer in enumerate(self.model):
    #activation of the layer will stored in x
    x=layer(x)
    #appending the activation of the selected layers and return the feature array
    if (str(layer_num) in self.req_features):
    features.append(x)             
    return features
```

- ## **Content features**
> 由於VGG卷積層能夠有效提取出各層特徵，並將圖像轉換為四維向量層層堆疊形成features map，接著進一步定義content loss為原圖與content image的均方誤差(MSE)，而這裡之所以採用MSE的理由是希望計算如下

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*PKnjB3bxzgg6yy0uOsljqw.png)

其中的content image 會預先採用複製原圖的方式，並迭代更新content loss。

實作代碼如下:
```
def calc_content_loss(gen_feat,orig_feat):
    content_l=torch.mean((gen_feat-orig_feat)**2) #*0.5
    return content_l
```


- ## **Style features**
> 在計算圖像的style時，採用餘弦相似性(Cosine similarity)來計算圖像本身的"相似性"，若將圖像的各特徵向量化後，那麼要評估任意向量間的相似度會變得非常有用。因此，當考慮任意兩向量在向量空間中，可透過計算向量的內積來知道，當兩向量成90度時，內積為零，意味此兩向量彼此毫無相關。

![](https://miro.medium.com/v2/resize:fit:490/format:webp/1*H1UW3bwrhqkRUJ11Xg6gGA.png)
(引用自參考資料[1])

![](https://ithelp.ithome.com.tw/upload/images/20230731/20158010TDpRsRF5Mt.png)(引用自參考資料[7])


> 將上述提及的cosine similarity推廣到圖像處理，相當於進一步計算圖像的特徵相關性分布，而這個分布形成的二維方陣稱作格拉姆矩陣(Gram matrix)，細節可參考[6]
> 這裡提到的gram matrix是指針對圖像在不同通道、像素下(nw,nh,nc)進行相關性(correlation)的計算，也就是說從Gram matrix中的數值大小，能夠看出合成圖和原圖在那些特徵的關係強弱，具體公式如下

![](https://miro.medium.com/v2/resize:fit:518/format:webp/1*C3fkQanKHMwOi_rf0q0OQQ.png)

> 對於style loss 的計算，同樣計算和原圖的均方誤差(MSE)估算變異量。代碼中使用torch.mm 將先前每一層所儲存的feature map進行矩陣相乘運算

> ![](https://ithelp.ithome.com.tw/upload/images/20230731/20158010ap1TLwzCOk.png)

- ## **Total Loss**
> 為了讓合成的圖樣產生最佳的效果，勢必在content loss和style loss間須取得平衡，因此分別引入α和β作為決定合成圖像中content 和style的成分多寡，在求解total loss 的最佳解過程採用梯度下降法(Gradient descent)搭配Adam優化器實現。

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*39DOPiFLq8TcncxuLKro7Q.png)

代碼中首先初始化loss後，去進一步原圖分別和content/style image的差異，最初由於
```
def calculate_loss(gen_features, orig_feautes, style_featues):
    style_loss=content_loss=0
    for gen,cont,style in zip(gen_features,orig_feautes,style_featues):
    content_loss+=calc_content_loss(gen,cont)
    style_loss+=calc_style_loss(gen,style)
    total_loss=alpha*content_loss + beta*style_loss 
    return total_loss
```

> 針對目前的結果來看，經過迭代更新1000次後，其實合成出來的圖象已經達到不錯的效果，從調整α和β的比例關係來決定原圖來自style的成分多寡，下圖展示了設置 α=1 β=10 的風格圖

![image](https://github.com/JunTingLu/neuron-style-transfer/assets/135250298/896620a7-4658-463e-81a5-8effc7661a77)



最後，完整代碼可參考操考資料[5]，歡迎互相交流，不吝指教~

### 參考資料
> 1. [Neural Networks Intuitions](https://towardsdatascience.com/neural-networks-intuitions-2-dot-product-gram-matrix-and-neural-style-transfer-5d39653e7916)
> 2. [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
> 3. [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)
> 4. [Deep Learning & Art: Neural Style Transfer](https://upscfever.com/upsc-fever/en/data/deeplearning4/Art+Generation+with+Neural+Style+Transfer+-+v2.html)
> 5. [github](https://github.com/JunTingLu/neuron-style-transfer/edit/main/README.md)
> 6. [Gram matrix](https://ccjou.wordpress.com/2011/03/07/%E7%89%B9%E6%AE%8A%E7%9F%A9%E9%99%A3-14%EF%BC%9Agramian-%E7%9F%A9%E9%99%A3/)
> 7. [格拉姆矩阵（Gram matrix）详细解读](https://www.cnblogs.com/yifanrensheng/p/12862174.html)
