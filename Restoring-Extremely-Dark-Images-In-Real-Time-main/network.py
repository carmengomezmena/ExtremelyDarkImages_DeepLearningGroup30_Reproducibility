import torch
from torch.nn.utils import weight_norm as wn
import torch.nn as nn

################### ICNR initialization for pixelshuffle        
def ICNR(tensor, upscale_factor=2, negative_slope=1, fan_type='fan_in'):
    # ICNR is an image segmentation method to separate an image according to objects location
    # Returns a kernel with kaiming_normal distributed values of the size of the input tensor
    
    # Int rounds down to next integer (1.9 -> 1)
    # tensor.shape[0] is an int size of the first dimension
    # list(tensor.shape[1:] is a list with the dimension of the tensor except for the first D
    # new_shape is a list with the size of each dimension of the tensor, with the first dimension adjusted by the upscale_factor
    new_shape = [int(tensor.shape[0] / (upscale_factor ** 2))] + list(tensor.shape[1:])
    subkernel = torch.zeros(new_shape)
    
    # kaiming_normal: function named after Kaiming He.
    # Fills the input tensor (subkernek) with a normal distribution N(0, std^2)
    # std = (gain / sqrt(fan/mode))
    # 'fan_in' preserves the magnitude of the weights in the fwd pass (and not in the backward pass?)
    nn.init.kaiming_normal_(subkernel, a=negative_slope, mode=fan_type, nonlinearity='leaky_relu')
    subkernel = subkernel.transpose(0, 1)
    
    # contiguous needs to be called to allocate memory of the transpose of a tensor the same way it would if the tensor was created in this shape
    # view returns a new tensor with the same value as the input
    subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                            subkernel.shape[1], -1)
    
    # Copies the kernel again, and scales it by the square of the upscale_factor
    kernel = subkernel.repeat(1, 1, upscale_factor ** 2)

    # Allocated new memory to kernel shaped like the initial tensor inputed
    # I don't know why they split the tensor dimension because they don't modify them at all
    transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)

    # Transpose the kernel
    kernel = kernel.transpose(0, 1)

    return kernel

def conv_layer(inc, outc, kernel_size=3, groups=1, bias=False, negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=4, num_classes=3, weight_normalization = True):

    layers = []
    
    if bn:
        m = nn.BatchNorm2d(inc)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        layers.append(m)
    
    if activation=='before':
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        
    if pixelshuffle_init:
        m = nn.Conv2d(in_channels=inc, out_channels=num_classes * (upscale ** 2),
                                  kernel_size=3, padding = 3//2, groups=1, bias=True, stride=1)
        nn.init.constant_(m.bias, 0)
        with torch.no_grad():
            kernel = ICNR(m.weight, upscale, negative_slope, fan_type)
            m.weight.copy_(kernel)
    else:
        m = nn.Conv2d(in_channels=inc, out_channels=outc,
     kernel_size=kernel_size, padding = (kernel_size-1)//2, groups=groups, bias=bias, stride=1)
        init_gain = 0.02
        if init_type == 'normal':
            torch.nn.init.normal_(m.weight, 0.0, init_gain)
        elif init_type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight, gain = init_gain)
        elif init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(m.weight, a=negative_slope, mode=fan_type, nonlinearity='leaky_relu')
        elif init_type == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    
    if weight_normalization:
        layers.append(wn(m))
    else:
        layers.append(m)
    
    if activation=='after':
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
            
    return nn.Sequential(*layers)
    
    
class ResBlock(nn.Module):    
    def __init__(self,in_c):
        super(ResBlock, self).__init__()
                
        self.conv1 = conv_layer(in_c, in_c, kernel_size=3, groups=1, bias=True, negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', activation='after', pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        
        self.conv2 = conv_layer(in_c, in_c, kernel_size=3, groups=1, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x
        
        
class make_dense(nn.Module):
    
    def __init__(self, nChannels=64, growthRate=32, pos=False):
        super(make_dense, self).__init__()
        
        kernel_size=3
        if pos=='first':
            self.conv = conv_layer(nChannels, growthRate, kernel_size=kernel_size, groups=1, bias=False, negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        elif pos=='middle':
            self.conv = conv_layer(nChannels, growthRate, kernel_size=kernel_size, groups=1, bias=False, negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', activation='before', pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        elif pos=='last':
            self.conv = conv_layer(nChannels, growthRate, kernel_size=kernel_size, groups=1, bias=False, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation='before', pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        else:
            raise NotImplementedError('ReLU position error in make_dense')
        
    def forward(self, x):
        return torch.cat((x, self.conv(x)), 1)
        
        
        
class RDB(nn.Module):
    def __init__(self, nChannels=96, nDenselayer=5, growthRate=32):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        
        modules.append(make_dense(nChannels_, growthRate, 'first'))
        nChannels_ += growthRate
        for i in range(nDenselayer-2):    
            modules.append(make_dense(nChannels_, growthRate, 'middle'))
            nChannels_ += growthRate 
        modules.append(make_dense(nChannels_, growthRate, 'last'))
        nChannels_ += growthRate
        
        self.dense_layers = nn.Sequential(*modules)
            
        self.conv_1x1 = conv_layer(nChannels_, nChannels, kernel_size=1, groups=1, bias=False, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
    
    def forward(self, x):
        return self.conv_1x1(self.dense_layers(x)) + x
        
        
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        
        self.up4 = nn.PixelShuffle(4)
        self.up2 = nn.PixelShuffle(2)
        
        self.conv32x = nn.Sequential(        
                        conv_layer(1024, 128, kernel_size=3, groups=128, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True),
                        conv_layer(128, 64, kernel_size=3, groups=1, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
                        )
                        
        self.RDB1 = RDB(nChannels=64, nDenselayer=4, growthRate=32)
        self.RDB2 = RDB(nChannels=64, nDenselayer=5, growthRate=32)
        self.RDB3 = RDB(nChannels=64, nDenselayer=5, growthRate=32)

        self.rdball = conv_layer(int(64*3), 64, kernel_size=1, groups=1, bias=False, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        
        self.conv_rdb8x = conv_layer(int(64//16), 64, kernel_size=3, groups=1, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        
        self.resblock8x = ResBlock(64)
        
        self.conv32_8_cat = nn.Sequential(
                        conv_layer(128, 32, kernel_size=3, groups=4, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True),
                        conv_layer(32, 192, kernel_size=3, groups=1, bias=True, negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', activation='after', pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True),
                        self.up4)                      
        
        
        self.conv2x = conv_layer(4, 12, kernel_size=5, groups=1, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        
        self.conv_2_8_32 = conv_layer(24, 12, kernel_size=5, groups=1, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        
        
    def downshuffle(self,var,r):
        b,c,h,w = var.size()
        out_channel = c*(r**2)
        out_h = h//r
        out_w = w//r
        return var.contiguous().view(b, c, out_h, r, out_w, r).permute(0,1,3,5,2,4).contiguous().view(b,out_channel, out_h, out_w).contiguous()
        
        
    def forward(self,low):
            
        low2x = self.downshuffle(low,2)
        
        # 32x branch starts
        low32x_beforeRDB = self.conv32x(self.downshuffle(low2x,16))
        rdb1 = self.RDB1(low32x_beforeRDB)
        rdb2 = self.RDB2(rdb1)
        rdb3 = self.RDB3(rdb2)
        rdb8x = torch.cat((rdb1,rdb2,rdb3),dim=1)
        rdb8x = self.rdball(rdb8x)+low32x_beforeRDB
        rdb8x = self.up4(rdb8x)
        rdb8x = self.conv_rdb8x(rdb8x)
        
        # 8x branch starts
        low8x = self.resblock8x(self.downshuffle(low2x,4))
        cat_32_8 = torch.cat((low8x,rdb8x),dim=1).contiguous()
        
        b,c,h,w = cat_32_8.size()
        G=2
        cat_32_8 = cat_32_8.view(b, G, c // G, h, w).permute(0, 2, 1, 3, 4).contiguous().view(b, c, h, w)
        cat_32_8 = self.conv32_8_cat(cat_32_8)
        
        
        # 2x branch starts
        low2x = torch.cat((self.conv2x(low2x),cat_32_8),dim=1)
        low2x = self.up2(self.conv_2_8_32(low2x))
        
        return torch.clamp(low2x,min=0.0, max=1.0)
