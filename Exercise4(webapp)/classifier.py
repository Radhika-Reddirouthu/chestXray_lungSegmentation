import torch
import torchvision
import pandas as pd
import numpy as np
from PIL import Image
from io import StringIO
import os
from keras.preprocessing import image
from keras.models import load_model

def allowed_file(filename):
    '''
    Checks if a given file `filename` is of type image with 'png', 'jpg', or 'jpeg' extensions
    '''
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'jfif'])
    return (('.' in filename) and (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS))

#Below are the classes and methods used by unet model to predict the lung segmentation
    
class Block(torch.nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels, batch_norm=False):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channel, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, padding=1)
        
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(mid_channel)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
            
    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = torch.nn.functional.relu(x, inplace=True)
        
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        out = torch.nn.functional.relu(x, inplace=True)
        return out
    
class UNet(torch.nn.Module):
    def up(self, x, size):
        return torch.nn.functional.interpolate(x, size=size, mode=self.upscale_mode)
    
    def down(self, x):
        return torch.nn.functional.max_pool2d(x, kernel_size=2)
    
    def __init__(self, in_channels, out_channels, batch_norm=False, upscale_mode="nearest"):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.upscale_mode = upscale_mode
        
        self.enc1 = Block(in_channels, 64, 64, batch_norm)
        self.enc2 = Block(64, 128, 128, batch_norm)
        self.enc3 = Block(128, 256, 256, batch_norm)
        self.enc4 = Block(256, 512, 512, batch_norm)
        
        self.center = Block(512, 1024, 512, batch_norm)
        
        self.dec4 = Block(1024, 512, 256, batch_norm)
        self.dec3 = Block(512, 256, 128, batch_norm)
        self.dec2 = Block(256, 128, 64, batch_norm)
        self.dec1 = Block(128, 64, 64, batch_norm)
        
        self.out = torch.nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.down(enc1))
        enc3 = self.enc3(self.down(enc2))
        enc4 = self.enc4(self.down(enc3))
        
        center = self.center(self.down(enc4))
        
        dec4 = self.dec4(torch.cat([self.up(center, enc4.size()[-2:]), enc4], 1))
        dec3 = self.dec3(torch.cat([self.up(dec4, enc3.size()[-2:]), enc3], 1))
        dec2 = self.dec2(torch.cat([self.up(dec3, enc2.size()[-2:]), enc2], 1))
        dec1 = self.dec1(torch.cat([self.up(dec2, enc1.size()[-2:]), enc1], 1))
        
        out = self.out(dec1)
        
        return out


def blend(origin, out):
    '''crop the predicted output with black background'''
    origin=torchvision.transforms.functional.to_pil_image(origin+0.5)
    out=torchvision.transforms.functional.to_pil_image(out+0.0)
    origin_pixel=list(origin.getdata())
    out_pixel=list(out.getdata())
    for i in range(len(out_pixel)):
        if out_pixel[i]==0:
            origin_pixel[i]=0
    origin.putdata(origin_pixel)
    origin=origin.convert("RGB")
    return origin

#predicts if a image is a chest xray or not and if it is a chest xray then predicts the lung area

def lung_segmentation(image_input):
    #load chest_xray or not binary model
    model=load_model("saved_models/chest_xray_binary.h5") #give path of the saved model(chest xray or not binary model) 
    #give the path of the static folder
    image_input=os.path.join('static', image_input)
    img = image.load_img(image_input, target_size=(200, 200))
    x=image.img_to_array(img)
    x = np.expand_dims(x/255., axis=0)
    images = np.vstack([x])
    classes = model.predict(images)
    if classes[0]>0.5:
        #load unet lung segmentation model
        models_path='saved_models/unet_model.pt' #give path of the saved model(lung segmentation model) 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        unet = UNet(in_channels=1, out_channels=2, batch_norm=True, upscale_mode="bilinear")
        unet.load_state_dict(torch.load(models_path, map_location=torch.device("cpu")))
        unet.to(device)
        unet.eval();
        origin = Image.open(image_input).convert("P")
        origin = torchvision.transforms.functional.resize(origin, (256, 256))
        origin = torchvision.transforms.functional.to_tensor(origin) - 0.5
        with torch.no_grad():
            origin = torch.stack([origin])
            origin = origin.to(device)
            out = unet(origin)
            softmax = torch.nn.functional.log_softmax(out, dim=1)
            out = torch.argmax(softmax, dim=1)
            
            origin = origin[0].to("cpu")
            out = out[0].to("cpu")
        pil_origin = torchvision.transforms.functional.to_pil_image(origin + 0.5).convert("RGB")
        output_image=np.array(blend(origin,out))
        output_image=Image.fromarray(output_image)
        output_image.save('static/output.png')
    else:
        img.save('static/output.png')





