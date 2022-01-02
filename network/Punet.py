import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_pconv import PConv2d
from network.pconv import PConv2d

class Pconv_lr(nn.Module):  # different padding from tf source code
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Pconv2d_bias = PConv2d(in_channels = in_channels, 
                                    out_channels = out_channels, 
                                    kernel_size = 3, 
                                    stride = 1, 
                                    # padding = 'valid', 
                                    padding = 0, 
                                    bias = True, 
        )
        
    def forward(self, x, mask):
        padding_shape = (1,1,1,1)
        x = F.pad(x, padding_shape, "replicate")
        mask = F.pad(mask, padding_shape, "constant", value=1)
        x, mask = self.Pconv2d_bias(x, mask)
        x = F.leaky_relu(x, negative_slope=0.1)
        return x, mask
    
    
class conv_lr(nn.Module):  # different padding from tf source code
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2d_bias = nn.Conv2d(in_channels = in_channels, 
                                     out_channels = out_channels, 
                                     kernel_size = 3,
                                     stride = 1, 
                                     padding = 'valid', 
                                     # padding = 'same', 
                                     bias = True, 
        )
        
    def forward(self, x, drop_rate=0.3):
        x = F.dropout(x, drop_rate)
        padding_shape = (1,1,1,1)
        x = F.pad(x, padding_shape, "replicate")
        x = self.conv2d_bias(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        return x
    
    
class conv(nn.Module):  # different padding from tf source code
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2d_bias = nn.Conv2d(in_channels = in_channels, 
                                     out_channels = out_channels, 
                                     kernel_size = 3,
                                     stride = 1, 
                                     padding = 'valid', 
                                     # padding = 'same', 
                                     bias = True, 
        )
        self.Sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x, drop_rate=0.3):
        x = F.dropout(x, drop_rate)
        padding_shape = (1,1,1,1)
        x = F.pad(x, padding_shape, "replicate")
        x = self.conv2d_bias(x)
        x = self.Sigmoid(x)
        return x
        

class Punet(nn.Module):
    def __init__(self, channel=3, width=256, height=256, drop_rate=0.3):
        super().__init__()
        
        self.channel = channel
        self.width = width
        self.height = height
        self.drop_rate = drop_rate

        # encoder
        self.env_conv0 = Pconv_lr(self.channel, 48)  # in_channel=x.channel, out_channel=output channel
        self.env_conv1 = Pconv_lr(48, 48)
        self.env_conv2 = Pconv_lr(48, 48)
        self.env_conv3 = Pconv_lr(48, 48)
        self.env_conv4 = Pconv_lr(48, 48)
        self.env_conv5 = Pconv_lr(48, 48)
        self.env_conv6 = Pconv_lr(48, 48)
        # decoder
        self.dec_conv5 = conv_lr(96, 96)
        self.dec_conv5b = conv_lr(96, 96)
        self.dec_conv4 = conv_lr(144, 96)
        self.dec_conv4b = conv_lr(96, 96)
        self.dec_conv3 = conv_lr(144, 96)
        self.dec_conv3b = conv_lr(96, 96)
        self.dec_conv2 = conv_lr(144, 96)
        self.dec_conv2b = conv_lr(96, 96)
        self.dec_conv1a = conv_lr(99, 64)
        self.dec_conv1b = conv_lr(64, 32)
        self.dec_conv1 = conv(32, self.channel)
    
    def Pmaxpool2d(self, x, mask, kernel_size=2):
        # pooling = nn.MaxPool2d(kernel_size=kernel_size, padding='same')
        pooling = nn.MaxPool2d(kernel_size=kernel_size)
        x = pooling(x)
        mask = pooling(mask)
        return x, mask
    
    def encoder(self, x, mask):
        skips = [x]
        
        x, mask = self.env_conv0(x, mask)
        x, mask = self.env_conv1(x, mask)
        x, mask = self.Pmaxpool2d(x, mask)
        skips.append(x)
        
        x, mask = self.env_conv2(x, mask)
        x, mask = self.Pmaxpool2d(x, mask)
        skips.append(x)
        
        x, mask = self.env_conv3(x, mask)
        x, mask = self.Pmaxpool2d(x, mask)
        skips.append(x)
        
        x, mask = self.env_conv4(x, mask)
        x, mask = self.Pmaxpool2d(x, mask)
        skips.append(x)
        
        x, mask = self.env_conv5(x, mask)
        x, mask = self.Pmaxpool2d(x, mask)
        x, mask = self.env_conv6(x, mask)
        
        return x, skips
    
    def decoder(self, x, skips):
        x = F.upsample(x, scale_factor=2)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec_conv5(x, self.drop_rate)
        x = self.dec_conv5b(x, self.drop_rate)
        
        x = F.upsample(x, scale_factor=2)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec_conv4(x, self.drop_rate)
        x = self.dec_conv4b(x, self.drop_rate)
        
        x = F.upsample(x, scale_factor=2)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec_conv3(x, self.drop_rate)
        x = self.dec_conv3b(x, self.drop_rate)
        
        x = F.upsample(x, scale_factor=2)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec_conv2(x, self.drop_rate)
        x = self.dec_conv2b(x, self.drop_rate)
        
        x = F.upsample(x, scale_factor=2)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec_conv1a(x, self.drop_rate)
        x = self.dec_conv1b(x, self.drop_rate)
        x = self.dec_conv1(x, self.drop_rate)
        return x 
    
    def forward(self, x, mask):
        x, skips = self.encoder(x, mask)
        x = self.decoder(x, skips)
        return x